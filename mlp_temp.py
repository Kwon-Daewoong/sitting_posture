"""
거북목/자세 판별 MLP — 16차원 입력 / 이진 분류 


[입력 데이터]
- posture_merged.csv (3개 CSV 평균, 591행)
- feature: right_eye/ear/shoulder/hip × (x, y, z, v) = 16차원
- label: label_binary (0=Good, 1=Bad)

[적용 기법]
1. 전처리: 어깨 기준 상대좌표 + StandardScaler
2. 증강: 가우시안 노이즈
3. 모델: MLP + BatchNorm + Dropout + He 초기화
4. 학습: AdamW + CosineAnnealingLR + Label Smoothing + Gradient Clipping
5. 검증: 5-Fold + Optuna 30회 탐색
6. 저장: Best val_acc 모델 + scaler + config
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import subprocess

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler


# ════════════════════════════════════════════════════════════════
# 1. 재현성 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")


# ════════════════════════════════════════════════════════════════
# 2. 설정값
CONFIG = {
    # 데이터
    "CSV_PATH":  "posture_merged.csv",
    "LABEL_COL": "label_binary",
    "FEATURE_COLS": [
        "right_eye_x","right_eye_y","right_eye_z","right_eye_v",
        "right_ear_x","right_ear_y","right_ear_z","right_ear_v",
        "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_v",
        "right_hip_x","right_hip_y","right_hip_z","right_hip_v",
    ],
    "OUTPUT_DIM": 2,

    # 16차원 인덱스
    "X_IDX": [0, 4, 8,  12],# 모든 x 좌표
    "Y_IDX": [1, 5, 9,  13],# 모든 y 좌표
    "Z_IDX": [2, 6, 10, 14],# 모든 z 좌표
    "V_IDX": [3, 7, 11, 15],# 모든 v (신뢰도) — 상대좌표 변환 X
    "SH_X_IDX": 8,# right_shoulder_x
    "SH_Y_IDX": 9,# right_shoulder_y
    "SH_Z_IDX": 10,# right_shoulder_z

    # 교차검증 (k-flod)
    "N_FOLDS": 5,

    # Optuna
    "USE_OPTUNA": True,
    "N_TRIALS": 30,

    # 학습
    "EPOCHS": 200,
    "EARLY_STOP_PATIENCE": 25,
    "GRAD_CLIP": 1.0,
    "LABEL_SMOOTHING": 0.1,

    # 증강
    "USE_AUGMENTATION": True,
    "AUG_NOISE_STD": 0.01,

    # 좌표 정규화
    "USE_RELATIVE_COORDS": True,

    # 저장 경로
    "MODEL_SAVE_PATH": "best_posture_mlp.pth",
    "RESULT_DIR":"results",
}

os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 3. 어깨 기준 상대좌표 변환
def to_relative_coords(X: np.ndarray) -> np.ndarray:
    """
    어깨 (right_shoulder)를 원점으로 하는 상대좌표 변환.
    - x, y, z: 어깨 좌표 기준으로 평행이동
    - v (신뢰도): 변환하지 않음

    효과: 사람이 화면 어디에 있든 자세 자체만 학습
    """
    X_rel = X.copy()
    sh_x = X[:, CONFIG["SH_X_IDX"]:CONFIG["SH_X_IDX"]+1]
    sh_y = X[:, CONFIG["SH_Y_IDX"]:CONFIG["SH_Y_IDX"]+1]
    sh_z = X[:, CONFIG["SH_Z_IDX"]:CONFIG["SH_Z_IDX"]+1]

    X_rel[:, CONFIG["X_IDX"]] -= sh_x
    X_rel[:, CONFIG["Y_IDX"]] -= sh_y
    X_rel[:, CONFIG["Z_IDX"]] -= sh_z
    # V_IDX는 그대로 유지 (신뢰도)

    return X_rel


def augment_sample(x: np.ndarray, noise_std: float) -> np.ndarray:
    """
    가우시안 노이즈 주입 — MediaPipe 좌표 흔들림 시뮬레이션.

    신뢰도 v에는 노이즈 미주입 (값이 0~1로 제한된 확률값이라).
    """
    '''
    #좌우대칭 : 사람 좌우 무관하게 학습 진행
    if np.random.rand() < flip_prob:
        x_aug[[0, 2, 4, 6]] *= -1  # 모든 x좌표
    '''
    x_aug = x.copy()
    # x, y, z에만 노이즈 추가
    coord_idx = CONFIG["X_IDX"] + CONFIG["Y_IDX"] + CONFIG["Z_IDX"]
    x_aug[coord_idx] += np.random.normal(0, noise_std, size=len(coord_idx))
    return x_aug



    '''
    차후 여기에 규칙기반 feature 추가 후 차원을 늘리면 됨 
    
    
    '''
# ════════════════════════════════════════════════════════════════
# 4.Dataset
class PostureDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.01):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment_sample(x, self.noise_std)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx])


# ════════════════════════════════════════════════════════════════
# 5. MLP 모델
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ════════════════════════════════════════════════════════════════
# 6. 학습 / 검증
def train_one_epoch(model, loader, criterion, optimizer, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct+= (out.argmax(1) == yb).sum().item()
        total+= len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    preds_all, labels_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = criterion(out, yb)
        total_loss += loss.item() * len(yb)
        preds = out.argmax(1)
        correct += (preds == yb).sum().item()
        total+= len(yb)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(yb.cpu().tolist())
    return total_loss / total, correct / total, np.array(preds_all), np.array(labels_all)


# ════════════════════════════════════════════════════════════════
# 7. 단일 학습 루프
def train_model(X_train, y_train, X_val, y_val, params, verbose=False):
    train_ds = PostureDataset(X_train, y_train,augment=CONFIG["USE_AUGMENTATION"], noise_std=CONFIG["AUG_NOISE_STD"])
    val_ds   = PostureDataset(X_val, y_val, augment=False)
    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False)

    model = PostureMLP(
        input_dim   = X_train.shape[1],
        hidden_dims = params["hidden_dims"],
        output_dim  = CONFIG["OUTPUT_DIM"],
        dropout_p   = params["dropout_p"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["LABEL_SMOOTHING"])
    optimizer = torch.optim.AdamW(model.parameters(),lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["EPOCHS"], eta_min=1e-6)

    best_acc, best_state, no_improve = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion,optimizer, CONFIG["GRAD_CLIP"])
        val_loss, val_acc, _, _ = evaluate(model, val_dl, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"Train: {tr_loss:.4f}/{tr_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f} | "
                  f"Best: {best_acc:.4f}")

        if no_improve >= CONFIG["EARLY_STOP_PATIENCE"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_acc, best_state, history


# ════════════════════════════════════════════════════════════════
# 8.Optuna 하이퍼파라미터 탐색
def objective(trial, X_tr, y_tr, X_val, y_val):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [
        trial.suggest_categorical(f"h{i}", [32, 64, 128, 256])
        for i in range(n_layers)
    ]
    params = {
        "hidden_dims":  hidden_dims,
        "dropout_p":    trial.suggest_float("dropout_p", 0.1, 0.5),
        "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    best_acc, _, _ = train_model(X_tr, y_tr, X_val, y_val, params)
    return best_acc


# ════════════════════════════════════════════════════════════════
# 9. K-Fold 교차검증
def run_kfold(X, y, best_params):
    skf = StratifiedKFold(n_splits=CONFIG["N_FOLDS"], shuffle=True, random_state=SEED)
    fold_results = []
    g_best_acc, g_best_state, g_best_scaler = 0.0, None, None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n━━━━━━━━━ Fold {fold}/{CONFIG['N_FOLDS']} ━━━━━━━━━")
        X_tr_raw, X_val_raw = X[tr_idx], X[val_idx]
        y_tr, y_val         = y[tr_idx], y[val_idx]

        # fold마다 scaler 새로 적합 (data leakage 방지)
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw)
        X_val  = scaler.transform(X_val_raw)

        best_acc, best_state, history = train_model(
            X_tr, y_tr, X_val, y_val, best_params, verbose=True)

        # 상세 평가
        val_ds = PostureDataset(X_val, y_val, augment=False)
        val_dl = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)
        model  = PostureMLP(
            input_dim   = X_tr.shape[1],
            hidden_dims = best_params["hidden_dims"],
            output_dim  = CONFIG["OUTPUT_DIM"],
            dropout_p   = best_params["dropout_p"],
        ).to(DEVICE)
        model.load_state_dict(best_state)
        _, val_acc, preds, labels = evaluate(model, val_dl, nn.CrossEntropyLoss())
        val_f1 = f1_score(labels, preds, average="weighted")

        print(f"  ▶ Fold {fold} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        fold_results.append({"fold": fold, "val_acc": val_acc,
                              "val_f1": val_f1, "history": history})

        if val_acc > g_best_acc:
            g_best_acc, g_best_state, g_best_scaler = val_acc, best_state, scaler

    print("\n" + "═" * 60)
    accs = [r["val_acc"] for r in fold_results]
    f1s  = [r["val_f1"]  for r in fold_results]
    print(f"K-Fold 결과:")
    print(f"  평균 Val Acc: {np.mean(accs):.4f} (±{np.std(accs):.4f})")
    print(f"  평균 Val F1 : {np.mean(f1s):.4f} (±{np.std(f1s):.4f})")
    print(f"  최고 Val Acc: {g_best_acc:.4f}")

    return g_best_state, g_best_scaler, fold_results


# ════════════════════════════════════════════════════════════════
# 10. 메인
def main():
    # 1. 데이터 로드 
    print("\n[Step 1] CSV 로드")
    df = pd.read_csv(CONFIG["CSV_PATH"], encoding="utf-8-sig")
    print(f"  전체 샘플: {len(df)}")
    print(f"  라벨 분포: Good={sum(df[CONFIG['LABEL_COL']]==0)} "
          f"/ Bad={sum(df[CONFIG['LABEL_COL']]==1)}")

    X = df[CONFIG["FEATURE_COLS"]].values.astype(np.float32)
    y = df[CONFIG["LABEL_COL"]].values.astype(np.int64)

    # 2. 상대좌표 변환
    if CONFIG["USE_RELATIVE_COORDS"]:
        print("\n[Step 2] 어깨 기준 상대좌표 변환")
        X = to_relative_coords(X)

    # 3. Optuna 탐색
    if CONFIG["USE_OPTUNA"]:
        print(f"\n[Step 3] Optuna 하이퍼파라미터 탐색 ({CONFIG['N_TRIALS']}회)")

        X_tr_o, X_val_o, y_tr_o, y_val_o = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        scaler_o = StandardScaler()
        X_tr_o   = scaler_o.fit_transform(X_tr_o)
        X_val_o  = scaler_o.transform(X_val_o)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",sampler=TPESampler(seed=SEED))
        study.optimize(
            lambda trial: objective(trial, X_tr_o, y_tr_o, X_val_o, y_val_o),
            n_trials=CONFIG["N_TRIALS"],
            show_progress_bar=True,
        )
        print(f"\n  ▶ 최고 Val Acc: {study.best_value:.4f}")
        print(f"  ▶ 최적 파라미터: {study.best_params}")

        n_layers = study.best_params["n_layers"]
        best_params = {
            "hidden_dims":[study.best_params[f"h{i}"] for i in range(n_layers)],
            "dropout_p":study.best_params["dropout_p"],
            "lr":study.best_params["lr"],
            "weight_decay": study.best_params["weight_decay"],
            "batch_size":study.best_params["batch_size"],
        }
    else:
        best_params = {
            "hidden_dims":[128, 64, 32],
            "dropout_p":0.3,
            "lr":1e-3,
            "weight_decay": 1e-4,
            "batch_size":32,
        }

    #4. K-Fold
    print(f"\n[Step 4] {CONFIG['N_FOLDS']}-Fold 교차검증")
    best_state, best_scaler, fold_results = run_kfold(X, y, best_params)

    #5. 모델 저장 
    print("\n[Step 5] 최종 모델 저장")
    torch.save({
        "model_state":best_state,
        "scaler_mean":best_scaler.mean_,
        "scaler_scale": best_scaler.scale_,
        "config":CONFIG,
        "best_params": best_params,
        "fold_results": [{k: v for k, v in r.items() if k != "history"}
                         for r in fold_results],
    }, CONFIG["MODEL_SAVE_PATH"])
    print(f"  저장 완료: {CONFIG['MODEL_SAVE_PATH']}")

    # ── 6. 시각화 ──
    fig, axes = plt.subplots(2, CONFIG["N_FOLDS"],figsize=(4*CONFIG["N_FOLDS"], 6))
    for i, r in enumerate(fold_results):
        h = r["history"]
        axes[0, i].plot(h["train_loss"], label="train")
        axes[0, i].plot(h["val_loss"], label="val")
        axes[0, i].set_title(f'Fold {r["fold"]} Loss')
        axes[0, i].legend()
        axes[1, i].plot(h["train_acc"], label="train")
        axes[1, i].plot(h["val_acc"], label="val")
        axes[1, i].set_title(f'Fold {r["fold"]} Acc')
        axes[1, i].legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['RESULT_DIR']}/training_curves.png", dpi=100)
    print(f"  학습 곡선 저장: {CONFIG['RESULT_DIR']}/training_curves.png")


# ════════════════════════════════════════════════════════════════
#11. 추론 함수
'''
def predict(landmarks_16, model_path=CONFIG["MODEL_SAVE_PATH"]): 
    ckpt = torch.load(model_path, map_location=DEVICE)

    model = PostureMLP(
        input_dim   = len(CONFIG["FEATURE_COLS"]),
        hidden_dims = ckpt["best_params"]["hidden_dims"],
        output_dim  = CONFIG["OUTPUT_DIM"],
        dropout_p   = ckpt["best_params"]["dropout_p"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = np.array(landmarks_16, dtype=np.float32).reshape(1, -1)
    if CONFIG["USE_RELATIVE_COORDS"]:
        x = to_relative_coords(x)
    x = (x - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
    x_t = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(x_t), dim=1)
        pred  = probs.argmax(1).item()
        conf  = probs[0, pred].item()

    return ("Good" if pred == 0 else "Bad"), conf
'''

if __name__ == "__main__":
    print("[Step 0] pre_csv.py 실행")

    result = subprocess.run(["python", "pre_csv.py"],capture_output=True,text=True)

    print(result.stdout)

    if result.returncode != 0:
        print("pre_csv.py 실행 실패")
        print(result.stderr)
        exit()
    main()