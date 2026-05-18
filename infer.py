"""
거북목/자세 판별 MLP — 추론 전용

1. 먼저 mlp_posture_final.py로 학습을 마쳐서 best_posture_mlp.pth 파일이 생성되어 있어야 함
2. 이 파일을 실행하면 저장된 모델을 불러와서 추론만 수행

[입력]
- 16개 좌표 리스트 (MediaPipe에서 미리 추출된 값)

[수정 사항]
-입력 : 16개 좌표 + 3개 파생 feature 
-판별 : 1차 학습기반 이진분류 + 2차 규칙기반(목/상체 각각 진단)
-출력 : 상태(정상/주의/위험) + 어디 문제 + 목각도 + 상체 각도 

[2단계 판별]
-cva각도 > 15도 -> 목문제 
-trunk_angle > 9도 -> 상체 문제 

[최종 판정]
 - mlp good + 둘다 정상 -> 정상
 - mlp bad + 하나만 문제 -> 주의 (어디문제인지)
 - mlp bad + 둘다 문제 -> 위험 
 - mlp bad + 둘다 정상 -> 주의(모델 이상 감지) 

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
# 설정 (학습 파일과 동일하게 유지)
# ════════════════════════════════════════════════════════════════
MODEL_PATH = "best_posture_mlp.pth"   # 학습 완료된 모델 파일 경로

# 16차원 좌표 인덱스 (학습 파일과 동일)
X_IDX = [0, 4, 8,  12]   # 모든 x 좌표
Y_IDX = [1, 5, 9,  13]   # 모든 y 좌표
Z_IDX = [2, 6, 10, 14]   # 모든 z 좌표
SH_X_IDX = 8             # right_shoulder_x
SH_Y_IDX = 9             # right_shoulder_y
SH_Z_IDX = 10            # right_shoulder_z

USE_RELATIVE_COORDS = True
FEATURE_DIM = 19
OUTPUT_DIM  = 2

#규칙 기반 판정 임계값추가 ㅣ 목 15, 상체 9  
CVA_THRESHOLD   = 15.0
TRUNK_THRESHOLD = 9.0   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════
# feature 계산 함수 
# ════════════════════════════════════════════════════════════════
def compute_derived_features(landmarks_16):
    """
    16개 좌표로부터 3개 파생 feature 계산.
    preprocess.py의 add_derived_features()와 동일 로직.
 
    Returns: (cva_angle, ear_shoulder_dist, trunk_angle)

    [수정]
    abs 제거 학습py 동일
    """
    # 좌표 추출 (인덱스 순서: eye, ear, shoulder, hip × x,y,z,v)
    ear_x, ear_y = landmarks_16[4], landmarks_16[5]
    sh_x,  sh_y  = landmarks_16[8], landmarks_16[9]
    hip_x, hip_y = landmarks_16[12], landmarks_16[13]
 
    # CVA 각도: 귀-어깨 선이 수직선과 이루는 각도
    cva_angle = np.degrees(np.arctan2(
        (ear_x - sh_x),
        (sh_y - ear_y)
    ))
 
    # 귀-어깨 거리
    ear_shoulder_dist = np.sqrt(
        (ear_x - sh_x) ** 2 + (ear_y - sh_y) ** 2
    )
 
    # 상체 기울기: 어깨-힙 선이 수직선과 이루는 각도
    trunk_angle = np.degrees(np.arctan2(
        (sh_x - hip_x),
        (hip_y - sh_y)
    ))
 
    return float(cva_angle), float(ear_shoulder_dist), float(trunk_angle)



# ════════════════════════════════════════════════════════════════
# 모델 클래스 (학습 파일과 동일하게 정의 필요)
# ════════════════════════════════════════════════════════════════
class PostureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════
# 전처리 함수 (학습 파일과 동일하게 유지)
# ════════════════════════════════════════════════════════════════
def to_relative_coords(X: np.ndarray) -> np.ndarray:
    """어깨 기준 상대좌표 변환"""
    X_rel = X.copy()
    sh_x = X[:, SH_X_IDX:SH_X_IDX+1]
    sh_y = X[:, SH_Y_IDX:SH_Y_IDX+1]
    sh_z = X[:, SH_Z_IDX:SH_Z_IDX+1]
    X_rel[:, X_IDX] -= sh_x
    X_rel[:, Y_IDX] -= sh_y
    X_rel[:, Z_IDX] -= sh_z
    return X_rel


# 모델로드 
def load_model(model_path: str = MODEL_PATH):
    """
    저장된 .pth 파일에서 모델 + scaler 정보를 불러옴
    여러 번 추론할 거면 이 함수는 한 번만 호출하고
    반환된 (model, ckpt) 튜플을 재사용
    """
    print(f"[모델 로드] {model_path}")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = PostureMLP(
        input_dim   = FEATURE_DIM,
        hidden_dims = ckpt["best_params"]["hidden_dims"],
        output_dim  = OUTPUT_DIM,
        dropout_p   = ckpt["best_params"]["dropout_p"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()   # 추론 모드 (Dropout/BatchNorm 비활성화)

    print(f"  모델 구조: {ckpt['best_params']['hidden_dims']}")
    print(f"  K-Fold 최고 정확도: "
          f"{max(r['val_acc'] for r in ckpt['fold_results']):.4f}")
    print("  로드 완료\n")

    return model, ckpt


#추론함수
def predict(landmarks_16, model, ckpt):
    """
    16개 좌표값 -> Good/Bad 판별 -> 2단계 판별 + 상세진단결과 
    """
    # 입력 검증
    if len(landmarks_16) != 16:
        raise ValueError(f"입력은 16개여야 함, 받은 개수: {len(landmarks_16)}")


    # feature 계산
    cva_angle, ear_shoulder_dist, trunk_angle = compute_derived_features(landmarks_16)

    #19차원으로 확장
    landmarks_19 = list(landmarks_16) + [cva_angle, ear_shoulder_dist, trunk_angle]
    
    # numpy 변환
    x = np.array(landmarks_19, dtype=np.float32).reshape(1, -1)

    # 전처리: 상대좌표 -> scaler
    if USE_RELATIVE_COORDS:
        x = to_relative_coords(x)
    x = (x - ckpt["scaler_mean"]) / ckpt["scaler_scale"]

    # 모델 추론 (1차 이진분류)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor)
        probs  = F.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
        #conf = probs[0, pred].item()
        prob_good = probs[0, 0].item()
        prob_bad  = probs[0, 1].item()

    mlp_result = "Good" if pred == 0 else "Bad"
    mlp_confidence = probs[0,pred].item()

    # 2차 : 규칙기반 판별 
    issues = []
    if cva_angle > CVA_THRESHOLD:
        issues.append("목 문제")
    if trunk_angle > TRUNK_THRESHOLD:
        issues.append("상체 문제")


    # 최종 상태 출력 
    if mlp_result == "Good" and len(issues) == 0:
        status = "정상"
    elif mlp_result == "Bad" and len(issues) >= 2:
        status = "위험"
    else:
        status = "주의"


    return {
        "status": status,
        "issues": issues,
        "mlp_result": mlp_result,
        "mlp_confidence": mlp_confidence,
        "prob_good": prob_good,
        "prob_bad": prob_bad,
        "cva_angle": cva_angle,
        "trunk_angle": trunk_angle,
        "ear_shoulder_dist": ear_shoulder_dist,
    }

# 출력 
def print_result(result: dict, sample_name: str = ""):
    """판별 결과 보기 좋게 출력"""
    status_marker = {
        "정상": "[정상]",
        "주의": "[주의]",
        "위험": "[위험]",
    }[result["status"]]
 
    print(f"\n{'='*50}")
    if sample_name:
        print(f"{sample_name}")
        print('-' * 50)
 
    print(f"  최종 상태: {status_marker} {result['status']}")
 
    if result["issues"]:
        print(f"  문제 부위: {', '.join(result['issues'])}")
    else:
        print(f"  문제 부위: 없음")
 
    print(f"\n  [1차] MLP 판정: {result['mlp_result']} "
          f"(신뢰도 {result['mlp_confidence']:.1%})")
    print(f"        Good 확률: {result['prob_good']:.1%}")
    print(f"        Bad  확률: {result['prob_bad']:.1%}")
 
    print(f"\n  [2차] 규칙 기반 진단:")
 
    # CVA(목) 표시
    cva_status = "OK" if result["cva_angle"] <= CVA_THRESHOLD else "NotGood"
    print(f"        목 각도 (CVA):    {result['cva_angle']:6.2f}° "
          f"(임계값 {CVA_THRESHOLD}°)  [{cva_status}]")
 
    # Trunk(상체) 표시
    trunk_status = "OK" if result["trunk_angle"] <= TRUNK_THRESHOLD else "NotGood"
    print(f"        상체 각도:        {result['trunk_angle']:6.2f}° "
          f"(임계값 {TRUNK_THRESHOLD}°)  [{trunk_status}]")
 
    print(f"        귀-어깨 거리:    {result['ear_shoulder_dist']:.4f}")
# 테스트 코드
def main():
    # 1. 모델 로드 (한 번만)
    model, ckpt = load_model()

    #아래는 임시 더미값 실제 사용 시 실제 데이터로 교체
    test_samples = [
        # 자세 샘플
        {
            "name": "샘플 A",
            "landmarks": [
                0.74, 0.02, -0.43, 1.00,
                0.70, 0.00, -0.33, 1.00,
                0.62, 0.20, -0.40, 1.00,
                0.51, 0.66, -0.17, 0.98,
            ],
        },
        # 다른 자세 예시
        {
            "name": "샘플 B",
            "landmarks": [
                0.78, 0.10, -0.20, 0.99,
                0.74, 0.08, -0.15, 0.99,
                0.60, 0.22, -0.30, 1.00,
                0.50, 0.68, -0.15, 0.97,
            ],
        },
        {
            "name": "Good Sample B",
            "landmarks": [
                0.50, 0.02, -0.02, 1.00,
                0.48, 0.03, -0.07, 1.00,
                0.45, 0.18, -0.20, 1.00,
                0.47, 0.61, -0.15, 1.00,
            ],
        }, 
    ]

    #추론 실행
    print("=" * 50)
    print("자세 판별 결과")
    print("=" * 50)
    for sample in test_samples:
        result = predict(sample["landmarks"], model, ckpt)
        print_result(result, sample["name"])


    

if __name__ == "__main__":
    main()