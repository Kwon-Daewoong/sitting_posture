"""
3개 CSV (신뢰도 0.3 / 0.5 / 0.7) 좌표 평균내서 통합 CSV 생성
================================================

같은 (Label, FileName) 이미지의 좌표를 평균내서 더 안정적인 데이터 생성.
이진 라벨 매핑: Good2/Good3 → 0, Bad1/Bad2 → 1
"""

import pandas as pd
import numpy as np

# ====================================================
# 1. 3개 CSV 로드
INPUT_FILES = ["posture_0.3.csv", "posture_0.5.csv", "posture_0.7.csv"]
OUTPUT_FILE = "posture_merged.csv"

dfs = []
for f in INPUT_FILES:
    df = pd.read_csv(f, encoding="utf-8-sig")
    df["source_threshold"] = f.split("_")[-1].replace(".csv", "")
    dfs.append(df)
    print(f"  {f}: {len(df)}행")

combined = pd.concat(dfs, ignore_index=True)
print(f"\n단순 합산: {len(combined)}행")

# ====================================================
# 2. (Label, FileName) 기준으로 좌표 평균
COORD_COLS = [
    "right_eye_x", "right_eye_y", "right_eye_z", "right_eye_v",
    "right_ear_x", "right_ear_y", "right_ear_z", "right_ear_v",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_v",
    "right_hip_x", "right_hip_y", "right_hip_z", "right_hip_v",
]

merged = (combined.groupby(["Label", "FileName"], as_index=False)[COORD_COLS].mean())

# 평균에 사용된 CSV 개수도 기록 (1~3)
count_per_image = (combined.groupby(["Label", "FileName"]).size().reset_index(name="n_sources"))
merged = merged.merge(count_per_image, on=["Label", "FileName"])

print(f"평균 후 고유 이미지: {len(merged)}행")
print(f"\n평균에 사용된 CSV 수 분포:")
print(merged["n_sources"].value_counts().sort_index())

# ====================================================
# 3. 이진 라벨 추가 (Good*/Bad*)
def to_binary_label(label_str: str) -> int:
    """Good2/Good3 → 0, Bad1/Bad2 → 1"""
    if label_str.startswith("Good"):
        return 0
    elif label_str.startswith("Bad"):
        return 1
    else:
        raise ValueError(f"알 수 없는 라벨: {label_str}")

merged["label_binary"] = merged["Label"].apply(to_binary_label)

print(f"\n원본 라벨 분포:")
print(merged["Label"].value_counts())
print(f"\n이진 라벨 분포:")
print(merged["label_binary"].value_counts())

# ====================================================
# 4. 저장
# 컬럼 순서 정리
output_cols = ["Label", "label_binary", "FileName"] + COORD_COLS + ["n_sources"]
merged = merged[output_cols]

merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {OUTPUT_FILE}")
print(f"최종 행 수: {len(merged)}")