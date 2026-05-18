"""
3개 CSV (신뢰도 0.3 / 0.5 / 0.7) 좌표 평균내서 통합 CSV 생성
================================================

같은 (Label, FileName) 이미지의 좌표를 평균내서 더 안정적인 데이터 생성.
이진 라벨 매핑: Good2/Good3 → 0, Bad1/Bad2 → 1

# 수정사항
- 뒤로 기운 자세를 위해서 feature 계산시 부호 살리기 
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


#파생 features  추가 
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    측면 자세 분석을 위한 3개 파생 Feature 계산.
 
    1) cva_angle (CVA: Craniovertebral Angle)
       - 귀-어깨 선과 수직선이 이루는 각도
       - 작을수록 정상, 클수록 거북목
 
    2) ear_shoulder_dist
       - 귀와 어깨 사이의 2D 직선 거리
       - 거북목일수록 귀가 어깨보다 앞으로 나와 거리가 커짐
 
    3) trunk_angle (상체 기울기)
       - 어깨-힙 선과 수직선이 이루는 각도
       - 작을수록곧은 상체 클수록 굽은 상체

    [Featur - 부호 살리기 ]
    cva_angle  > 0: 귀가 어깨보다 앞 -> 거북목
    cva_angle  < 0: 귀가 어깨보다 뒤 -> 뒤로 젖힘
    cva_angle = 0: 정상
    
    trunk_angle  > 0: 어깨가 힙보다 앞-> 상체 앞으로 굽음
    trunk_angle  < 0: 어깨가 힙보다 뒤 -> 상체 뒤로 젖힘
    trunk_angle = 0: 곧은 자세

    """
    ear_x, ear_y = df["right_ear_x"], df["right_ear_y"]
    sh_x,  sh_y  = df["right_shoulder_x"], df["right_shoulder_y"]
    hip_x, hip_y = df["right_hip_x"],      df["right_hip_y"]
 
    #CVA 각도
    df["cva_angle"] = np.degrees(np.arctan2(
        (ear_x - sh_x),
        (sh_y - ear_y)
    ))
 
    #귀-어깨 거리
    df["ear_shoulder_dist"] = np.sqrt(
        (ear_x - sh_x) ** 2 + (ear_y - sh_y) ** 2
    )
 
    #  상체 기울기
    df["trunk_angle"] = np.degrees(np.arctan2(
        (sh_x - hip_x),
        (hip_y - sh_y)
    ))
 
    return df
 
merged = add_derived_features(merged)
print(f"\n파생 Feature 추가 완료:")
print(merged[["cva_angle", "ear_shoulder_dist", "trunk_angle"]].describe().round(3))


# ====================================================
# Good / Bad Feature 평균 비교
# ====================================================

print("\n[Good vs Bad 평균 비교]")

print(merged.groupby("label_binary")[["cva_angle", "ear_shoulder_dist", "trunk_angle"]]
      .mean().round(3))
 
print("\n[Good vs Bad 중앙값 비교]")
print(merged.groupby("label_binary")[["cva_angle", "ear_shoulder_dist", "trunk_angle"]]
      .median().round(3))
 
#부호별 분포 확인
print("\n[CVA 부호별 분포 — 앞으로/뒤로 구분]")
merged["cva_direction"] = merged["cva_angle"].apply(
    lambda x: "앞으로" if x > 15 else "뒤로" if x < -15 else "정상범위"
)
print(pd.crosstab(merged["Label"], merged["cva_direction"]))


# Good / Bad Feature 중앙값 비교
print("\n[Good vs Bad 중앙값 비교]")

print(
    merged.groupby("label_binary")[
        [
            "cva_angle",
            "ear_shoulder_dist",
            "trunk_angle"
        ]
    ]
    .median()
    .round(3)
)


# CVA 이상치 확인 TOP10
print("\n[CVA 이상치 TOP10]")

print(
    merged.sort_values(
        "cva_angle",
        ascending=False
    )[
        [
            "FileName",
            "Label",
            "cva_angle"
        ]
    ]
    .head(10)
)

# Trunk Angle 이상치 확인 TOP10
print("\n[Trunk Angle 이상치 TOP10]")

print(
    merged.sort_values(
        "trunk_angle",
        ascending=False
    )[
        [
            "FileName",
            "Label",
            "trunk_angle"
        ]
    ]
    .head(10)
)

# ====================================================
# 4. 저장
# 컬럼 순서 정리 + feature 추가 
DERIVED_COLS = ["cva_angle", "ear_shoulder_dist", "trunk_angle"]
output_cols = (
    ["Label", "label_binary", "FileName"]
    + COORD_COLS
    + DERIVED_COLS
    + ["n_sources"]
)

#output_cols = ["Label", "label_binary", "FileName"] + COORD_COLS + ["n_sources"]
merged = merged[output_cols]

merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {OUTPUT_FILE}")
print(f"최종 행 수: {len(merged)}")