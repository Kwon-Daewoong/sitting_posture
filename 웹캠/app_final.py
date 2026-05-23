import time
from collections import deque

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from infer import load_model, predict


# =========================
# Streamlit UI 설정
# =========================
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


# =========================
# 기본 설정
# =========================
VISIBILITY_THRESHOLD = 0.5

# Temporal Logic 설정
FRAME_WINDOW_SIZE = 10
RATIO_THRESHOLD = 0.7
WARNING_SECONDS = 5

# 각도 임계값: infer.py와 동일하게 유지
CVA_THRESHOLD = 15.0
TRUNK_THRESHOLD = 9.0

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


# =========================
# 모델 로드
# =========================
@st.cache_resource
def load_posture_model():
    return load_model("best_posture_mlp.pth")


model, ckpt = load_posture_model()


# =========================
# Video Processor
# =========================
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

        # 최근 프레임의 3단계 상태("정상", "주의", "위험") 저장
        self.status_buffer = deque(maxlen=FRAME_WINDOW_SIZE)
        self.danger_start_time = None

    def extract_landmarks_16(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, results, None

        landmarks = results.pose_landmarks.landmark

        right_score = sum(landmarks[idx].visibility for idx in RIGHT_JOINTS.values())
        left_score = sum(landmarks[idx].visibility for idx in LEFT_JOINTS.values())

        if left_score > right_score:
            selected = LEFT_JOINTS
            side = "left"
        else:
            selected = RIGHT_JOINTS
            side = "right"

        is_valid = all(
            landmarks[idx].visibility >= VISIBILITY_THRESHOLD
            for idx in selected.values()
        )

        if not is_valid:
            return None, results, side

        features = []

        for idx in selected.values():
            lm = landmarks[idx]

            # 학습 데이터가 right 기준이면 left를 right처럼 맞추기 위해 x 반전
            x = 1.0 - lm.x if side == "left" else lm.x
            features.extend([x, lm.y, lm.z, lm.visibility])

        return features, results, side

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
        """
        회의 기준 3단계 판정:
        - CVA 정상 + Trunk 정상 -> 정상(GOOD)
        - 둘 중 하나만 위험 -> 주의(CAUTION)
        - 둘 다 위험 -> 위험(DANGER)
        """
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
        """
        3단계 상태("정상", "주의", "위험")에 대해 temporal smoothing 적용.

        최근 N프레임 중
        - 위험 비율이 기준 이상이면 최종 위험
        - 비정상(주의+위험) 비율이 기준 이상이면 최종 주의
        - 그 외 정상
        """
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

        return temporal_status, danger_ratio, abnormal_ratio, warning_on

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        display_img = img.copy()

        # 1. landmark 추출
        landmarks_16, results, side = self.extract_landmarks_16(img)

        # 2. 전체 skeleton overlay
        if results is not None and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                display_img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        # 3. 프로젝트 핵심선: ear-shoulder-hip 별도 표시
        display_img = self.draw_selected_posture_line(display_img, results, side)

        if landmarks_16 is None:
            cv2.putText(
                display_img,
                "Pose not detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        # 4. MLP 추론 및 각도 계산
        result = predict(landmarks_16, model, ckpt)

        # 5. CVA / Trunk 각도 기준 3단계 판정
        angle_status, cva_bad, trunk_bad = self.classify_by_angles(result)

        # 6. Temporal Logic
        temporal_status, danger_ratio, abnormal_ratio, warning_on = self.apply_temporal_logic(angle_status)

        # 색상 설정
        if temporal_status == "정상":
            color = (0, 255, 0)
            status_text = "GOOD"
        elif temporal_status == "주의":
            color = (0, 255, 255)
            status_text = "CAUTION"
        else:
            color = (0, 0, 255)
            status_text = "DANGER"

        # 화면 표시
        cv2.putText(
            display_img,
            f"Status: {status_text}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        cv2.putText(
            display_img,
            f"MLP: {result['mlp_result']} ({result['mlp_confidence']:.2f})",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            display_img,
            f"CVA: {result['cva_angle']:.2f} ({'BAD' if cva_bad else 'OK'})",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            display_img,
            f"Trunk: {result['trunk_angle']:.2f} ({'BAD' if trunk_bad else 'OK'})",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            display_img,
            f"Danger Ratio: {danger_ratio:.2f}",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            display_img,
            f"Abnormal Ratio: {abnormal_ratio:.2f}",
            (30, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        issue_texts = []
        if cva_bad:
            issue_texts.append("Neck")
        if trunk_bad:
            issue_texts.append("Trunk")

        if issue_texts:
            cv2.putText(
                display_img,
                "Issue: " + ", ".join(issue_texts),
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        if warning_on:
            cv2.putText(
                display_img,
                "WARNING: FIX YOUR POSTURE",
                (30, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
            )

        return av.VideoFrame.from_ndarray(display_img, format="bgr24")


# =========================
# WebRTC 실행
# =========================
webrtc_streamer(
    key="posture-webcam",
    video_processor_factory=PostureVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
