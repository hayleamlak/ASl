import cv2
import mediapipe as mp
import numpy as np


def normalize_landmarks(flat_landmarks: np.ndarray) -> np.ndarray:
    reshaped = flat_landmarks.reshape(-1, 3)
    wrist = reshaped[0]
    centered = reshaped - wrist

    scale = np.max(np.linalg.norm(centered[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0

    centered = centered / scale
    return centered.flatten()


class HandLandmarkExtractor:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._hands_solution = mp.solutions.hands
        self._hands = self._hands_solution.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils

    def extract(self, frame_bgr: np.ndarray, draw: bool = True) -> tuple[np.ndarray | None, np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)

        annotated = frame_bgr.copy() if draw else frame_bgr
        if not result.multi_hand_landmarks:
            return None, annotated

        hand = result.multi_hand_landmarks[0]
        if draw:
            self._drawer.draw_landmarks(
                annotated,
                hand,
                self._hands_solution.HAND_CONNECTIONS,
            )

        flat = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32).flatten()
        return normalize_landmarks(flat), annotated

    def close(self) -> None:
        self._hands.close()
