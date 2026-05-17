"""
거북목/자세 판별 MLP — 추론 전용

1. 먼저 mlp_posture_final.py로 학습을 마쳐서 best_posture_mlp.pth 파일이 생성되어 있어야 함
2. 이 파일을 실행하면 저장된 모델을 불러와서 추론만 수행

[입력]
- 16개 좌표 리스트 (MediaPipe에서 미리 추출된 값)

[출력]
- "Good" 또는 "Bad" 판별 결과
- 신뢰도 (0~100%)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
# ① 설정 (학습 파일과 동일하게 유지)
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
FEATURE_DIM = 16
OUTPUT_DIM  = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════
# ② 모델 클래스 (학습 파일과 동일하게 정의 필요)
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
# ③ 전처리 함수 (학습 파일과 동일하게 유지)
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
    16개 좌표값 -> Good/Bad 판별
    """
    # 입력 검증
    if len(landmarks_16) != FEATURE_DIM:
        raise ValueError(f"입력은 {FEATURE_DIM}개여야 함, 받은 개수: {len(landmarks_16)}")

    # numpy 변환
    x = np.array(landmarks_16, dtype=np.float32).reshape(1, -1)

    # 전처리: 상대좌표 -> scaler
    if USE_RELATIVE_COORDS:
        x = to_relative_coords(x)
    x = (x - ckpt["scaler_mean"]) / ckpt["scaler_scale"]

    # 모델 추론
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor)
        probs  = F.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
        conf = probs[0, pred].item()
        prob_good = probs[0, 0].item()
        prob_bad  = probs[0, 1].item()

    result = "Good" if pred == 0 else "Bad"
    return result, conf, (prob_good, prob_bad)


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
    ]

    #추론 실행
    print("=" * 50)
    print("자세 판별 결과")
    print("=" * 50)
    for sample in test_samples:
        result, conf, (p_good, p_bad) = predict(
            sample["landmarks"], model, ckpt
        )
        print(f"\n[{sample['name']}]")
        print(f"  판별 결과: {result}")
        print(f"  신뢰도:    {conf:.2%}")
        print(f"  Good 확률: {p_good:.2%}")
        print(f"  Bad  확률: {p_bad:.2%}")


if __name__ == "__main__":
    main()