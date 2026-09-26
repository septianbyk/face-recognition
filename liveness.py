import time
from collections import deque
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path("face_landmarker.task")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def ensure_model():
    if not MODEL_PATH.exists():
        urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class BlinkLivenessDetector:
    def __init__(self, ear_threshold=0.21, liveness_window_seconds=4.0):
        model_path = ensure_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.ear_threshold = ear_threshold
        self.liveness_window_seconds = liveness_window_seconds
        self.blink_timestamps = deque()
        self.was_closed = False
        self.start_time = time.time()

    def update(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return False

        landmarks = result.face_landmarks[0]
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        now = time.time()
        is_closed = avg_ear < self.ear_threshold
        if is_closed and not self.was_closed:
            self.was_closed = True
        elif not is_closed and self.was_closed:
            self.blink_timestamps.append(now)
            self.was_closed = False

        while self.blink_timestamps and now - self.blink_timestamps[0] > self.liveness_window_seconds:
            self.blink_timestamps.popleft()

        return len(self.blink_timestamps) >= 1

    def reset(self):
        self.blink_timestamps.clear()
        self.was_closed = False