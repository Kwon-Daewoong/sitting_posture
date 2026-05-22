import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import sys

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

current_dir = os.getcwd()
BASE_TRAIN_PATH = os.path.join(current_dir, "Sitting_Posture_multiclass", "train")
RIGHT_JOINTS = {'eye': 2, 'ear': 8, 'shoulder': 12, 'hip': 24}
LEFT_SHOULDER_IDX = 11

def print_progress(iteration, total, prefix='', suffix='', length=40):
    # 터미널 진행 표시 바 출력
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '=' * filled_length + '>' + '-' * (length - filled_length - 1)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def extract_landmarks_raw(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results.pose_landmarks.landmark if results.pose_landmarks else None

def process_with_threshold(all_images_landmarks, threshold):
    processed_data = []
    for img_name, label, landmarks in all_images_landmarks:
        # 설정 신뢰도 이상 여부 확인
        is_valid = all(landmarks[idx].visibility >= threshold for idx in RIGHT_JOINTS.values())
        if is_valid:
            # 좌측면 판별 및 X축 대칭 변환
            is_left_side = landmarks[LEFT_SHOULDER_IDX].visibility > landmarks[RIGHT_JOINTS['shoulder']].visibility
            row_data = [img_name, label]
            for idx in RIGHT_JOINTS.values():
                lm = landmarks[idx]
                x = (1.0 - lm.x) if is_left_side else lm.x
                row_data.extend([x, lm.y, lm.z, lm.visibility])
            processed_data.append(row_data)
    return processed_data

def main():
    if not os.path.exists(BASE_TRAIN_PATH):
        print("경로 탐색 불가.")
        return

    image_files = [f for f in os.listdir(BASE_TRAIN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_imgs = len(image_files)
    print(f"총 {total_imgs}개 이미지 로드 완료.")

    raw_landmarks_pool = []
    for i, img_name in enumerate(image_files):
        label = img_name.split(' ')[0] if ' ' in img_name else "unknown"
        lms = extract_landmarks_raw(os.path.join(BASE_TRAIN_PATH, img_name))
        if lms:
            raw_landmarks_pool.append((img_name, label, lms))
        
        # 10개 단위 진행상황 업데이트
        if (i + 1) % 10 == 0 or (i + 1) == total_imgs:
            print_progress(i + 1, total_imgs, prefix='이미지 처리', suffix=f'({i+1}/{total_imgs})')

    print("\n\n신뢰도(0.1~0.9)별 전체 데이터셋 추출 시작.")
    
    thresholds = [round(x * 0.1, 1) for x in range(1, 10)]
    columns = ['file_name', 'label']
    for j in ['r_eye', 'r_ear', 'r_shoulder', 'r_hip']:
        columns.extend([f'{j}_x', f'{j}_y', f'{j}_z', f'{j}_v'])

    for ts in thresholds:
        # 신뢰도별 필터링
        filtered_data = process_with_threshold(raw_landmarks_pool, ts)
        
        if not filtered_data:
            print(f"신뢰도 {ts}: 유효 데이터 부족.")
            continue

        # 데이터프레임 생성 및 CSV 저장
        df = pd.DataFrame(filtered_data, columns=columns)
        output_name = f"all_landmarks_ts{int(ts*10)}.csv"
        df.to_csv(output_name, index=False, encoding='utf-8-sig')
        print(f"신뢰도 {ts} 완료: {len(df)}개 추출 ({output_name})")

    print("\n작업 완료.")

if __name__ == "__main__":
    main()
