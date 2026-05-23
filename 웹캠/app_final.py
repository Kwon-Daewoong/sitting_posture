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

            # 학습 데이터가 right 기준이면 left를 right처럼 맞추기 위해 x 반전
            x = 1.0 - lm.x if side == "left" else lm.x
            features.extend([x, lm.y, lm.z, lm.visibility])

        return features, results, side, "OK"

    def draw_text_panel(self, img, status_text, cva_text, trunk_text, warning_text=None, color=(255, 255, 255)):
        """
        최종 시연용 화면:
        Status, CVA, Trunk, Warning 중심으로 간단히 표시한다.
        """
        panel_x, panel_y = 20, 25
        line_gap = 38

        cv2.putText(
            img,
            f"Status: {status_text}",
            (panel_x, panel_y + line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        cv2.putText(
            img,
            cva_text,
            (panel_x, panel_y + line_gap * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

        cv2.putText(
            img,
            trunk_text,
            (panel_x, panel_y + line_gap * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

        if warning_text:
            cv2.putText(
                img,
                warning_text,
                (panel_x, panel_y + line_gap * 4 + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )

        return img

    def draw_side_view_guide(self, img):
        """
        측면 자세 안내 문구와 간단한 기준선을 표시한다.
        OpenCV는 한글 출력이 깨질 수 있어 영문으로 표시한다.
        """
        h, w, _ = img.shape

        guide_text = "SIDE VIEW ONLY - Keep ear, shoulder, and hip visible"
        cv2.putText(
            img,
            guide_text,
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # 사용자가 상체를 화면 중앙에 맞추도록 돕는 약한 세로 가이드라인
        x_center = int(w * 0.5)
        cv2.line(img, (x_center, 0), (x_center, h), (80, 80, 80), 1)

        return img

    def draw_measurement_unavailable(self, img, reason):
        """
        landmark가 불안정할 때 모델 추론을 수행하지 않고
        자세 조정 안내를 출력한다.
        """
        if reason == "NO_POSE":
            message1 = "Measurement unavailable"
            message2 = "No pose detected. Move into the camera frame."
        elif reason == "LOW_VISIBILITY":
            message1 = "Measurement unavailable"
            message2 = "Ear/shoulder/hip visibility is low. Adjust side position."
        else:
            message1 = "Measurement unavailable"
            message2 = "Please adjust your posture and camera position."

        cv2.putText(
            img,
            message1,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            message2,
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

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

        return temporal_status, warning_on

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        display_img = img.copy()

        # 측면 자세 안내 문구 및 가이드라인 표시
        display_img = self.draw_side_view_guide(display_img)

        # 1. landmark 추출
        landmarks_16, results, side, detect_status = self.extract_landmarks_16(img)

        # 2. 전체 skeleton overlay
        if results is not None and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                display_img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        # 3. 프로젝트 핵심선: ear-shoulder-hip 별도 표시
        display_img = self.draw_selected_posture_line(display_img, results, side)

        # landmark가 불안정하면 모델 추론하지 않고 측정 불가 안내
        if landmarks_16 is None:
            display_img = self.draw_measurement_unavailable(display_img, detect_status)
            return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        # 4. MLP 추론 및 각도 계산
        result = predict(landmarks_16, model, ckpt)

        # 5. CVA / Trunk 각도 기준 3단계 판정
        angle_status, cva_bad, trunk_bad = self.classify_by_angles(result)

        # 6. Temporal Logic
        temporal_status, warning_on = self.apply_temporal_logic(angle_status)

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

        cva_text = f"CVA: {result['cva_angle']:.2f} deg ({'BAD' if cva_bad else 'OK'})"
        trunk_text = f"Trunk: {result['trunk_angle']:.2f} deg ({'BAD' if trunk_bad else 'OK'})"
        warning_text = "WARNING: FIX YOUR POSTURE" if warning_on else None

        # 최종 시연용 화면: Status, CVA, Trunk, Warning 중심으로 정리
        display_img = self.draw_text_panel(
            display_img,
            status_text=status_text,
            cva_text=cva_text,
            trunk_text=trunk_text,
            warning_text=warning_text,
            color=color,
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
