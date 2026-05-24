"""
posture_detection_final.py

MediaPipe 기반 실시간 거북목 및 자세 판별 시스템
-------------------------------------------------
본 파일은 딥러닝 팀 프로젝트 제출용 단일 소스 코드 파일입니다.

GitHub 저장소:
https://github.com/Kwon-Daewoong/sitting_posture.git

============================================================
1. Git 저장소에 포함되어야 하는 주요 파일
============================================================

[필수 코드 파일]
- posture_detection_final.py
  또는 본 파일명인 posture_detection_final_no_sklearn_app.py
  → 전처리, 학습, 추론, Streamlit 웹캠 서비스를 하나로 합친 단일 소스 코드

[웹캠 서비스 실행에 필요한 파일]
- best_posture_mlp.pth
  → 학습 완료된 MLP 모델 파일
  → Streamlit 웹캠 실행 시 반드시 같은 폴더에 있어야 함

[train 모드 실행에 필요한 CSV 파일]
- posture_0.3.csv
- posture_0.5.csv
- posture_0.7.csv
  → 이 3개 파일이 있으면 train 모드에서 posture_merged.csv를 자동 생성함

[train 모드 실행 후 생성되는 파일]
- posture_merged.csv
  → 0.3 / 0.5 / 0.7 CSV를 평균내어 만든 최종 학습용 CSV
- best_posture_mlp.pth
  → 학습 완료 후 저장되는 모델 파일
- results/
  → 학습 결과 저장용 폴더

[선택 파일]
- README.md
  → 실행 방법, 환경 설정, 모델 파일 다운로드 방법 설명
- .gitignore
  → __pycache__, .DS_Store, 가상환경, 대용량 원본 데이터 등을 제외

============================================================
2. 설치해야 하는 패키지
============================================================

[웹캠 서비스(app) 실행만 할 경우]
pip install streamlit streamlit-webrtc opencv-python mediapipe torch av numpy pandas

[학습(train)까지 실행할 경우]
pip install streamlit streamlit-webrtc opencv-python mediapipe torch av numpy pandas scikit-learn optuna

[권장 Python 버전]
Python 3.11 권장
MediaPipe의 mp.solutions.pose API를 사용하므로 Python 3.13 환경에서는 버전 문제가 발생할 수 있음.

예시 conda 환경:
conda create -n posture python=3.11 -y
conda activate posture
pip install streamlit streamlit-webrtc opencv-python mediapipe torch av numpy pandas scikit-learn optuna

============================================================
3. 실행 방법
============================================================

[1] 학습 실행
python posture_detection_final.py --mode train

필요 파일:
- posture_0.3.csv
- posture_0.5.csv
- posture_0.7.csv

생성 파일:
- posture_merged.csv
- best_posture_mlp.pth


[2] 추론 테스트 실행
python posture_detection_final.py --mode infer

필요 파일:
- best_posture_mlp.pth


[3] Streamlit 웹캠 서비스 실행
streamlit run posture_detection_final.py -- --mode app

필요 파일:
- best_posture_mlp.pth

주의:
- best_posture_mlp.pth는 본 파일과 같은 폴더에 있어야 함
- 웹캠 서비스는 측면 자세를 기준으로 동작함
- 화면 위 텍스트는 OpenCV 한글 출력 문제를 피하기 위해 영어로 표시함

============================================================
4. 제출 관련 메모
============================================================

과제 조건이 "소스 코드 단일 파일 형태 제출"인 경우,
본 파일 하나를 제출하면 전체 코드 흐름을 확인할 수 있음.

다만 실제 실행을 위해서는 best_posture_mlp.pth 모델 파일이 필요하므로,
보고서 또는 README에 GitHub 저장소 주소를 명시하여 교수자가 모델 파일을 받을 수 있도록 함.

GitHub 저장소:
https://github.com/Kwon-Daewoong/sitting_posture.git
"""

import argparse
import os
import random
import time
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from streamlit_webrtc import webrtc_streamer
from torch.utils.data import DataLoader, Dataset

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    TPESampler = None


# ============================================================
# 공통 설정
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "best_posture_mlp.pth"
MERGED_CSV_PATH = "posture_merged.csv"
RESULT_DIR = "results"

BASE_FEATURE_COLS = [
    "right_eye_x", "right_eye_y", "right_eye_z", "right_eye_v",
    "right_ear_x", "right_ear_y", "right_ear_z", "right_ear_v",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_v",
    "right_hip_x", "right_hip_y", "right_hip_z", "right_hip_v",
]

DERIVED_FEATURE_COLS = [
    "cva_angle",
    "ear_shoulder_dist",
    "trunk_angle",
]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS

LABEL_COL = "label_binary"
FEATURE_DIM = 19
OUTPUT_DIM = 2

X_IDX = [0, 4, 8, 12]
Y_IDX = [1, 5, 9, 13]
Z_IDX = [2, 6, 10, 14]
SH_X_IDX = 8
SH_Y_IDX = 9
SH_Z_IDX = 10

USE_RELATIVE_COORDS = True

CVA_THRESHOLD = 15.0
TRUNK_THRESHOLD = 9.0

VISIBILITY_THRESHOLD = 0.5
FRAME_WINDOW_SIZE = 10
RATIO_THRESHOLD = 0.7
WARNING_SECONDS = 5

RIGHT_JOINTS = {
    "eye": 2,
    "ear": 8,
    "shoulder": 12,
    "hip": 24,
}

LEFT_JOINTS = {
    "eye": 5,
    "ear": 7,
    "shoulder": 11,
    "hip": 23,
}


# ============================================================
# 1. 데이터 전처리: posture_0.3/0.5/0.7.csv 평균 CSV 생성
# ============================================================
def to_binary_label(label_str: str) -> int:
    """Good* -> 0, Bad* -> 1"""
    if str(label_str).startswith("Good"):
        return 0
    if str(label_str).startswith("Bad"):
        return 1
    raise ValueError(f"알 수 없는 라벨: {label_str}")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    16개 좌표에서 파생 feature 3개 계산.
    - cva_angle: 귀-어깨 선이 수직선과 이루는 각도
    - ear_shoulder_dist: 귀와 어깨 사이 2D 거리
    - trunk_angle: 어깨-힙 선이 수직선과 이루는 각도
    """
    ear_x, ear_y = df["right_ear_x"], df["right_ear_y"]
    sh_x, sh_y = df["right_shoulder_x"], df["right_shoulder_y"]
    hip_x, hip_y = df["right_hip_x"], df["right_hip_y"]

    df["cva_angle"] = np.degrees(np.arctan2((ear_x - sh_x), (sh_y - ear_y)))
    df["ear_shoulder_dist"] = np.sqrt((ear_x - sh_x) ** 2 + (ear_y - sh_y) ** 2)
    df["trunk_angle"] = np.degrees(np.arctan2((sh_x - hip_x), (hip_y - sh_y)))

    return df


def build_merged_csv(input_files=None, output_file: str = MERGED_CSV_PATH) -> pd.DataFrame:
    """
    0.3 / 0.5 / 0.7 신뢰도 CSV를 평균내어 posture_merged.csv 생성.
    """
    if input_files is None:
        input_files = ["posture_0.3.csv", "posture_0.5.csv", "posture_0.7.csv"]

    dfs = []
    for file_name in input_files:
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"{file_name} 파일이 없습니다.")
        df = pd.read_csv(file_name, encoding="utf-8-sig")
        df["source_threshold"] = file_name.split("_")[-1].replace(".csv", "")
        dfs.append(df)
        print(f"[CSV 로드] {file_name}: {len(df)}행")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[합산] 총 {len(combined)}행")

    merged = combined.groupby(["Label", "FileName"], as_index=False)[BASE_FEATURE_COLS].mean()

    count_per_image = (
        combined.groupby(["Label", "FileName"])
        .size()
        .reset_index(name="n_sources")
    )
    merged = merged.merge(count_per_image, on=["Label", "FileName"])
    merged["label_binary"] = merged["Label"].apply(to_binary_label)
    merged = add_derived_features(merged)

    output_cols = ["Label", "label_binary", "FileName"] + BASE_FEATURE_COLS + DERIVED_FEATURE_COLS + ["n_sources"]
    merged = merged[output_cols]

    merged.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] {output_file}: {len(merged)}행")

    return merged


# ============================================================
# 2. MLP 학습
# ============================================================
def to_relative_coords(X: np.ndarray) -> np.ndarray:
    """right_shoulder를 원점으로 하는 상대좌표 변환."""
    X_rel = X.copy()
    sh_x = X[:, SH_X_IDX:SH_X_IDX + 1]
    sh_y = X[:, SH_Y_IDX:SH_Y_IDX + 1]
    sh_z = X[:, SH_Z_IDX:SH_Z_IDX + 1]

    X_rel[:, X_IDX] -= sh_x
    X_rel[:, Y_IDX] -= sh_y
    X_rel[:, Z_IDX] -= sh_z

    return X_rel


TRAIN_CONFIG = {
    "CSV_PATH": MERGED_CSV_PATH,
    "LABEL_COL": LABEL_COL,
    "FEATURE_COLS": FEATURE_COLS,
    "OUTPUT_DIM": OUTPUT_DIM,
    "N_FOLDS": 5,
    "USE_OPTUNA": True,
    "N_TRIALS": 30,
    "EPOCHS": 200,
    "EARLY_STOP_PATIENCE": 25,
    "GRAD_CLIP": 1.0,
    "LABEL_SMOOTHING": 0.1,
    "USE_AUGMENTATION": True,
    "AUG_NOISE_STD": 0.025,
    "AUG_TRANSLATE_RANGE": 0.05,
    "AUG_TRANSLATE_PROB": 0.5,
    "AUG_ROTATION_DEG": 5.0,
    "AUG_ROTATION_PROB": 0.5,
    "AUG_SCALE_RANGE": (0.95, 1.05),
    "AUG_SCALE_PROB": 0.5,
    "USE_RELATIVE_COORDS": True,
    "MODEL_SAVE_PATH": MODEL_PATH,
    "RESULT_DIR": RESULT_DIR,
}


def augment_sample(x: np.ndarray) -> np.ndarray:
    """좌표 기반 데이터 증강: 노이즈, 이동, 회전, 스케일."""
    x_aug = x.copy()

    coord_idx = X_IDX + Y_IDX + Z_IDX
    x_aug[coord_idx] += np.random.normal(0, TRAIN_CONFIG["AUG_NOISE_STD"], size=len(coord_idx))

    if np.random.rand() < TRAIN_CONFIG["AUG_TRANSLATE_PROB"]:
        tx = np.random.uniform(-TRAIN_CONFIG["AUG_TRANSLATE_RANGE"], TRAIN_CONFIG["AUG_TRANSLATE_RANGE"])
        ty = np.random.uniform(-TRAIN_CONFIG["AUG_TRANSLATE_RANGE"], TRAIN_CONFIG["AUG_TRANSLATE_RANGE"])
        x_aug[X_IDX] += tx
        x_aug[Y_IDX] += ty

    if np.random.rand() < TRAIN_CONFIG["AUG_ROTATION_PROB"]:
        angle_rad = np.radians(np.random.uniform(-TRAIN_CONFIG["AUG_ROTATION_DEG"], TRAIN_CONFIG["AUG_ROTATION_DEG"]))
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for xi, yi in zip(X_IDX, Y_IDX):
            x_old = x_aug[xi]
            y_old = x_aug[yi]
            x_aug[xi] = x_old * cos_a - y_old * sin_a
            x_aug[yi] = x_old * sin_a + y_old * cos_a

    if np.random.rand() < TRAIN_CONFIG["AUG_SCALE_PROB"]:
        scale = np.random.uniform(*TRAIN_CONFIG["AUG_SCALE_RANGE"])
        x_aug[X_IDX] *= scale
        x_aug[Y_IDX] *= scale
        x_aug[Z_IDX] *= scale

    return x_aug


class PostureDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = augment_sample(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx])


class PostureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_p):
        super().__init__()
        layers = []
        prev = input_dim

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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


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
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)

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
        preds = out.argmax(1)

        total_loss += loss.item() * len(yb)
        correct += (preds == yb).sum().item()
        total += len(yb)

        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(yb.cpu().tolist())

    return total_loss / total, correct / total, np.array(preds_all), np.array(labels_all)


def train_model(X_train, y_train, X_val, y_val, params, verbose=False):
    train_ds = PostureDataset(X_train, y_train, augment=TRAIN_CONFIG["USE_AUGMENTATION"])
    val_ds = PostureDataset(X_val, y_val, augment=False)

    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    model = PostureMLP(
        input_dim=X_train.shape[1],
        hidden_dims=params["hidden_dims"],
        output_dim=TRAIN_CONFIG["OUTPUT_DIM"],
        dropout_p=params["dropout_p"],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG["LABEL_SMOOTHING"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_CONFIG["EPOCHS"],
        eta_min=1e-6,
    )

    best_acc, best_state, no_improve = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, TRAIN_CONFIG["EPOCHS"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_dl, criterion, optimizer, TRAIN_CONFIG["GRAD_CLIP"]
        )
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
            print(
                f"  Epoch {epoch:3d} | "
                f"Train Loss/Acc: {tr_loss:.4f}/{tr_acc:.4f} | "
                f"Val Loss/Acc: {val_loss:.4f}/{val_acc:.4f} | "
                f"Best: {best_acc:.4f}"
            )

        if no_improve >= TRAIN_CONFIG["EARLY_STOP_PATIENCE"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_acc, best_state, history


def objective(trial, X_tr, y_tr, X_val, y_val):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [
        trial.suggest_categorical(f"h{i}", [32, 64, 128, 256])
        for i in range(n_layers)
    ]

    params = {
        "hidden_dims": hidden_dims,
        "dropout_p": trial.suggest_float("dropout_p", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }

    best_acc, _, _ = train_model(X_tr, y_tr, X_val, y_val, params)
    return best_acc


def run_kfold(X, y, best_params):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=TRAIN_CONFIG["N_FOLDS"], shuffle=True, random_state=SEED)

    fold_results = []
    global_best_acc = 0.0
    global_best_state = None
    global_best_scaler = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{TRAIN_CONFIG['N_FOLDS']}]")

        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        best_acc, best_state, history = train_model(
            X_train, y_train, X_val, y_val, best_params, verbose=True
        )

        val_ds = PostureDataset(X_val, y_val, augment=False)
        val_dl = DataLoader(val_ds, batch_size=best_params["batch_size"], shuffle=False)

        model = PostureMLP(
            input_dim=X_train.shape[1],
            hidden_dims=best_params["hidden_dims"],
            output_dim=TRAIN_CONFIG["OUTPUT_DIM"],
            dropout_p=best_params["dropout_p"],
        ).to(DEVICE)
        model.load_state_dict(best_state)

        _, val_acc, preds, labels = evaluate(model, val_dl, nn.CrossEntropyLoss())
        val_f1 = f1_score(labels, preds, average="weighted")

        print(f"  Fold {fold} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        fold_results.append({
            "fold": fold,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "history": history,
        })

        if val_acc > global_best_acc:
            global_best_acc = val_acc
            global_best_state = best_state
            global_best_scaler = scaler

    accs = [r["val_acc"] for r in fold_results]
    f1s = [r["val_f1"] for r in fold_results]

    print("\n[K-Fold 결과]")
    print(f"  평균 Val Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  평균 Val F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  최고 Val Acc: {global_best_acc:.4f}")

    return global_best_state, global_best_scaler, fold_results


def train_pipeline():
    """CSV 전처리 후 MLP 학습 및 모델 저장."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    os.makedirs(RESULT_DIR, exist_ok=True)

    if not os.path.exists(MERGED_CSV_PATH):
        print("[전처리] posture_merged.csv가 없어 새로 생성합니다.")
        build_merged_csv()

    print("\n[데이터 로드]")
    df = pd.read_csv(MERGED_CSV_PATH, encoding="utf-8-sig")
    print(f"  전체 샘플: {len(df)}")
    print(f"  라벨 분포: Good={sum(df[LABEL_COL] == 0)} / Bad={sum(df[LABEL_COL] == 1)}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.int64)

    if TRAIN_CONFIG["USE_RELATIVE_COORDS"]:
        print("[전처리] 어깨 기준 상대좌표 변환")
        X = to_relative_coords(X)

    if TRAIN_CONFIG["USE_OPTUNA"]:
        if optuna is None:
            raise ImportError("Optuna가 설치되어 있지 않습니다. pip install optuna 후 다시 실행하세요.")

        print(f"\n[Optuna 탐색] {TRAIN_CONFIG['N_TRIALS']}회")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED
        )

        scaler_o = StandardScaler()
        X_tr = scaler_o.fit_transform(X_tr)
        X_val = scaler_o.transform(X_val)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
        study.optimize(
            lambda trial: objective(trial, X_tr, y_tr, X_val, y_val),
            n_trials=TRAIN_CONFIG["N_TRIALS"],
            show_progress_bar=True,
        )

        print(f"  최고 Val Acc: {study.best_value:.4f}")
        print(f"  최적 파라미터: {study.best_params}")

        n_layers = study.best_params["n_layers"]
        best_params = {
            "hidden_dims": [study.best_params[f"h{i}"] for i in range(n_layers)],
            "dropout_p": study.best_params["dropout_p"],
            "lr": study.best_params["lr"],
            "weight_decay": study.best_params["weight_decay"],
            "batch_size": study.best_params["batch_size"],
        }
    else:
        best_params = {
            "hidden_dims": [128, 64, 32],
            "dropout_p": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 32,
        }

    best_state, best_scaler, fold_results = run_kfold(X, y, best_params)

    torch.save({
        "model_state": best_state,
        "scaler_mean": best_scaler.mean_,
        "scaler_scale": best_scaler.scale_,
        "config": TRAIN_CONFIG,
        "best_params": best_params,
        "fold_results": [
            {k: v for k, v in r.items() if k != "history"}
            for r in fold_results
        ],
    }, MODEL_PATH)

    print(f"\n[모델 저장 완료] {MODEL_PATH}")


# ============================================================
# 3. 추론 함수
# ============================================================
def compute_derived_features_from_landmarks(landmarks_16):
    """웹캠에서 추출한 16개 좌표로부터 파생 feature 계산."""
    if len(landmarks_16) != 16:
        raise ValueError(f"입력은 16개여야 합니다. 받은 개수: {len(landmarks_16)}")

    ear_x, ear_y = landmarks_16[4], landmarks_16[5]
    sh_x, sh_y = landmarks_16[8], landmarks_16[9]
    hip_x, hip_y = landmarks_16[12], landmarks_16[13]

    cva_angle = np.degrees(np.arctan2((ear_x - sh_x), (sh_y - ear_y)))
    ear_shoulder_dist = np.sqrt((ear_x - sh_x) ** 2 + (ear_y - sh_y) ** 2)
    trunk_angle = np.degrees(np.arctan2((sh_x - hip_x), (hip_y - sh_y)))

    return float(cva_angle), float(ear_shoulder_dist), float(trunk_angle)


def load_saved_model(model_path: str = MODEL_PATH):
    """저장된 best_posture_mlp.pth 로드."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} 파일이 없습니다. GitHub 저장소에서 모델 파일을 내려받아 같은 폴더에 두세요.\n"
            f"Repository: https://github.com/Kwon-Daewoong/sitting_posture.git"
        )

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = PostureMLP(
        input_dim=FEATURE_DIM,
        hidden_dims=ckpt["best_params"]["hidden_dims"],
        output_dim=OUTPUT_DIM,
        dropout_p=ckpt["best_params"]["dropout_p"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt


def predict_posture(landmarks_16, model, ckpt):
    """16개 좌표값 -> 파생 feature 추가 -> MLP 이진분류 -> 규칙 기반 상세 진단."""
    cva_angle, ear_shoulder_dist, trunk_angle = compute_derived_features_from_landmarks(landmarks_16)

    landmarks_19 = list(landmarks_16) + [cva_angle, ear_shoulder_dist, trunk_angle]
    x = np.array(landmarks_19, dtype=np.float32).reshape(1, -1)

    if USE_RELATIVE_COORDS:
        x = to_relative_coords(x)

    x = (x - ckpt["scaler_mean"]) / ckpt["scaler_scale"]
    x_tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1).item()
        prob_good = probs[0, 0].item()
        prob_bad = probs[0, 1].item()

    mlp_result = "Good" if pred == 0 else "Bad"
    mlp_confidence = probs[0, pred].item()

    issues = []
    if cva_angle > CVA_THRESHOLD:
        issues.append("Forward Head")
    elif cva_angle < -CVA_THRESHOLD:
        issues.append("Neck Leaned Back")

    if trunk_angle > TRUNK_THRESHOLD:
        issues.append("Trunk Forward")
    elif trunk_angle < -TRUNK_THRESHOLD:
        issues.append("Trunk Backward")

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


def infer_test():
    """간단한 더미 샘플 추론 테스트."""
    model, ckpt = load_saved_model()

    test_samples = [
        {
            "name": "Sample A",
            "landmarks": [
                0.74, 0.02, -0.43, 1.00,
                0.70, 0.00, -0.33, 1.00,
                0.62, 0.20, -0.40, 1.00,
                0.51, 0.66, -0.17, 0.98,
            ],
        },
        {
            "name": "Sample B",
            "landmarks": [
                0.50, 0.02, -0.02, 1.00,
                0.48, 0.03, -0.07, 1.00,
                0.45, 0.18, -0.20, 1.00,
                0.47, 0.61, -0.15, 1.00,
            ],
        },
    ]

    for sample in test_samples:
        result = predict_posture(sample["landmarks"], model, ckpt)
        print("\n" + "=" * 50)
        print(sample["name"])
        print(f"최종 상태: {result['status']}")
        print(f"MLP 판정: {result['mlp_result']} ({result['mlp_confidence']:.2%})")
        print(f"CVA: {result['cva_angle']:.2f}")
        print(f"Trunk: {result['trunk_angle']:.2f}")
        print(f"Issues: {', '.join(result['issues']) if result['issues'] else 'None'}")


# ============================================================
# 4. Streamlit 웹캠 서비스
# ============================================================
@st.cache_resource
def load_posture_model_for_app():
    return load_saved_model(MODEL_PATH)


class PostureVideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.status_buffer = deque(maxlen=FRAME_WINDOW_SIZE)
        self.danger_start_time = None

    def extract_landmarks_16(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, results, None, "NO_POSE"

        landmarks = results.pose_landmarks.landmark

        right_score = sum(landmarks[idx].visibility for idx in RIGHT_JOINTS.values())
        left_score = sum(landmarks[idx].visibility for idx in LEFT_JOINTS.values())

        if left_score > right_score:
            selected = LEFT_JOINTS
            side = "left"
        else:
            selected = RIGHT_JOINTS
            side = "right"

        low_visibility_joints = [
            name for name, idx in selected.items()
            if landmarks[idx].visibility < VISIBILITY_THRESHOLD
        ]

        if low_visibility_joints:
            return None, results, side, "LOW_VISIBILITY"

        features = []
        for idx in selected.values():
            lm = landmarks[idx]
            x = 1.0 - lm.x if side == "left" else lm.x
            features.extend([x, lm.y, lm.z, lm.visibility])

        return features, results, side, "OK"

    def draw_text_panel(self, img, status_text, cva_text, trunk_text, warning_text=None, color=(255, 255, 255)):
        panel_x, panel_y = 20, 25
        line_gap = 38

        cv2.putText(img, f"Status: {status_text}", (panel_x, panel_y + line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, cva_text, (panel_x, panel_y + line_gap * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(img, trunk_text, (panel_x, panel_y + line_gap * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        if warning_text:
            cv2.putText(img, warning_text, (panel_x, panel_y + line_gap * 4 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        return img

    def draw_side_view_guide(self, img):
        h, w, _ = img.shape

        guide_text = "SIDE VIEW ONLY - Keep ear, shoulder, and hip visible"
        cv2.putText(img, guide_text, (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        x_center = int(w * 0.5)
        cv2.line(img, (x_center, 0), (x_center, h), (80, 80, 80), 1)

        return img

    def draw_measurement_unavailable(self, img, reason):
        if reason == "NO_POSE":
            message1 = "Measurement unavailable"
            message2 = "No pose detected. Move into the camera frame."
        elif reason == "LOW_VISIBILITY":
            message1 = "Measurement unavailable"
            message2 = "Ear/shoulder/hip visibility is low. Adjust side position."
        else:
            message1 = "Measurement unavailable"
            message2 = "Please adjust your posture and camera position."

        cv2.putText(img, message1, (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(img, message2, (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        return img

    def draw_selected_posture_line(self, img, results, side):
        if results is None or not results.pose_landmarks or side is None:
            return img

        h, w, _ = img.shape
        landmarks = results.pose_landmarks.landmark
        selected = LEFT_JOINTS if side == "left" else RIGHT_JOINTS

        ear = landmarks[selected["ear"]]
        shoulder = landmarks[selected["shoulder"]]
        hip = landmarks[selected["hip"]]

        points = []
        for lm in [ear, shoulder, hip]:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))

        ear_pt, shoulder_pt, hip_pt = points

        cv2.circle(img, ear_pt, 7, (255, 0, 0), -1)
        cv2.circle(img, shoulder_pt, 7, (0, 255, 255), -1)
        cv2.circle(img, hip_pt, 7, (0, 255, 0), -1)

        cv2.line(img, ear_pt, shoulder_pt, (255, 255, 0), 3)
        cv2.line(img, shoulder_pt, hip_pt, (255, 255, 0), 3)

        cv2.putText(img, "ear", (ear_pt[0] + 5, ear_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(img, "shoulder", (shoulder_pt[0] + 5, shoulder_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(img, "hip", (hip_pt[0] + 5, hip_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

    def classify_by_angles(self, result):
        cva_angle = result["cva_angle"]
        trunk_angle = result["trunk_angle"]

        cva_bad = abs(cva_angle) > CVA_THRESHOLD
        trunk_bad = abs(trunk_angle) > TRUNK_THRESHOLD

        if cva_bad and trunk_bad:
            status = "위험"
        elif cva_bad or trunk_bad:
            status = "주의"
        else:
            status = "정상"

        return status, cva_bad, trunk_bad

    def apply_temporal_logic(self, angle_status):
        self.status_buffer.append(angle_status)

        n = len(self.status_buffer)
        danger_count = sum(1 for x in self.status_buffer if x == "위험")
        caution_count = sum(1 for x in self.status_buffer if x == "주의")
        abnormal_count = danger_count + caution_count

        danger_ratio = danger_count / n
        abnormal_ratio = abnormal_count / n

        if n < FRAME_WINDOW_SIZE:
            temporal_status = angle_status
        else:
            if danger_ratio >= RATIO_THRESHOLD:
                temporal_status = "위험"
            elif abnormal_ratio >= RATIO_THRESHOLD:
                temporal_status = "주의"
            else:
                temporal_status = "정상"

        now = time.time()

        if temporal_status == "위험":
            if self.danger_start_time is None:
                self.danger_start_time = now

            elapsed = now - self.danger_start_time
            warning_on = elapsed >= WARNING_SECONDS
        else:
            self.danger_start_time = None
            warning_on = False

        return temporal_status, warning_on

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        display_img = img.copy()

        display_img = self.draw_side_view_guide(display_img)

        landmarks_16, results, side, detect_status = self.extract_landmarks_16(img)

        if results is not None and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                display_img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        display_img = self.draw_selected_posture_line(display_img, results, side)

        if landmarks_16 is None:
            display_img = self.draw_measurement_unavailable(display_img, detect_status)
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        result = predict_posture(landmarks_16, APP_MODEL, APP_CKPT)
        angle_status, cva_bad, trunk_bad = self.classify_by_angles(result)
        temporal_status, warning_on = self.apply_temporal_logic(angle_status)

        if temporal_status == "정상":
            color = (0, 255, 0)
            status_text = "GOOD"
        elif temporal_status == "주의":
            color = (0, 255, 255)
            status_text = "CAUTION"
        else:
            color = (0, 0, 255)
            status_text = "DANGER"

        cva_text = f"CVA: {result['cva_angle']:.2f} deg ({'BAD' if cva_bad else 'OK'})"
        trunk_text = f"Trunk: {result['trunk_angle']:.2f} deg ({'BAD' if trunk_bad else 'OK'})"
        warning_text = "WARNING: FIX YOUR POSTURE" if warning_on else None

        display_img = self.draw_text_panel(
            display_img,
            status_text=status_text,
            cva_text=cva_text,
            trunk_text=trunk_text,
            warning_text=warning_text,
            color=color,
        )

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")


def run_streamlit_app():
    global APP_MODEL, APP_CKPT

    st.set_page_config(page_title="Posture Detection", layout="centered")

    st.title("실시간 거북목 / 자세 판별 시스템")
    st.write("MediaPipe landmark와 MLP 모델을 이용해 측면 자세를 실시간으로 판별합니다.")
    st.info("카메라를 사용자 측면에 두고, 귀·어깨·상체가 화면에 보이도록 앉아주세요.")

    st.markdown(
        """
        **판별 흐름**  
        웹캠 프레임 → MediaPipe Pose → 16개 landmark feature → MLP 추론 → CVA/Trunk 각도 판정 → Temporal Logic → 최종 상태 출력
        """
    )

    APP_MODEL, APP_CKPT = load_posture_model_for_app()

    webrtc_streamer(
        key="posture-webcam",
        video_processor_factory=PostureVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


# ============================================================
# 5. 실행 진입점
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "app"],
        default="app",
        help="실행 모드 선택: train / infer / app",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train_pipeline()
    elif args.mode == "infer":
        infer_test()
    elif args.mode == "app":
        run_streamlit_app()
