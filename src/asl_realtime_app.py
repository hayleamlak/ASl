import argparse
from collections import Counter, deque
import datetime as dt
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.asl_core.hand_landmarks import HandLandmarkExtractor
from src.asl_core.model_utils import load_labels, load_model, predict_label


def append_label(text: str, label: str) -> str:
    normalized = label.strip().upper()
    if normalized in {"SPACE", "_", "[SPACE]"}:
        if not text or text.endswith(" "):
            return text
        return text + " "
    if normalized in {"DEL", "BACKSPACE"}:
        return text[:-1]
    if normalized in {"BLANK", "NOTHING", "NONE"}:
        return text
    return text + label


def get_last_word(text: str) -> str:
    parts = text.rstrip().split()
    if not parts:
        return ""
    return parts[-1]


def get_smoothed_label(buffer: deque[str], min_votes: int) -> tuple[str | None, int]:
    if not buffer:
        return None, 0

    label, votes = Counter(buffer).most_common(1)[0]
    if votes < min_votes:
        return None, votes
    return label, votes


def draw_panel(
    frame,
    text: str,
    current_label: str,
    conf: float,
    threshold: float,
    smoothed_label: str,
    votes: int,
    current_word: str,
) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 160), (20, 20, 20), -1)
    cv2.putText(frame, "ASL -> Text (q=quit, c=clear, b=backspace, s=save)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, f"Detected: {current_label} ({conf:.2f})  threshold={threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Smoothed: {smoothed_label}  votes={votes}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Word: {current_word[-20:]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 255), 2)
    cv2.putText(frame, f"Text: {text[-70:]}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 120), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time ASL to text using webcam landmarks.")
    parser.add_argument("--model", default="models/asl_model.joblib", help="Path to trained model (.joblib)")
    parser.add_argument("--labels", default="models/labels.txt", help="Path to labels file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum confidence to accept prediction")
    parser.add_argument("--stable-frames", type=int, default=6, help="Consecutive frames needed before commit")
    parser.add_argument("--smooth-window", type=int, default=9, help="Prediction window size used for majority-vote smoothing")
    parser.add_argument("--smooth-min-votes", type=int, default=5, help="Minimum votes for a smoothed label")
    parser.add_argument("--cooldown", type=float, default=0.7, help="Seconds between committed characters")
    parser.add_argument("--auto-space-delay", type=float, default=1.2, help="Seconds without a hand before inserting automatic space")
    parser.add_argument("--save-path", default="outputs/live_predictions.txt", help="Output file for saved text")
    args = parser.parse_args()

    _ = load_labels(args.labels)
    model = load_model(args.model)

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    final_text = ""
    candidate_label = ""
    candidate_count = 0
    label_window: deque[str] = deque(maxlen=max(1, args.smooth_window))
    last_commit_time = 0.0
    last_hand_seen_time = time.time()
    auto_space_armed = False
    current_label = "None"
    current_conf = 0.0
    smoothed_label = "None"
    smoothed_votes = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            features, annotated = extractor.extract(frame, draw=True)
            if features is not None:
                last_hand_seen_time = time.time()
                auto_space_armed = True
                label, confidence = predict_label(model, features)
                current_label = label
                current_conf = confidence

                if confidence >= args.threshold:
                    label_window.append(label)
                    voted_label, votes = get_smoothed_label(label_window, args.smooth_min_votes)
                    smoothed_votes = votes
                    if voted_label is None:
                        smoothed_label = "None"
                        candidate_count = 0
                        candidate_label = ""
                    else:
                        smoothed_label = voted_label
                        if voted_label == candidate_label:
                            candidate_count += 1
                        else:
                            candidate_label = voted_label
                            candidate_count = 1

                        if candidate_count >= args.stable_frames and (time.time() - last_commit_time) >= args.cooldown:
                            final_text = append_label(final_text, candidate_label)
                            last_commit_time = time.time()
                            candidate_count = 0
                            candidate_label = ""
                            label_window.clear()
                else:
                    label_window.clear()
                    smoothed_label = "None"
                    smoothed_votes = 0
                    candidate_count = 0
                    candidate_label = ""
            else:
                current_label = "No hand"
                current_conf = 0.0
                label_window.clear()
                smoothed_label = "None"
                smoothed_votes = 0
                candidate_count = 0
                candidate_label = ""

                if auto_space_armed and (time.time() - last_hand_seen_time) >= args.auto_space_delay:
                    if final_text and not final_text.endswith(" "):
                        final_text += " "
                    auto_space_armed = False

            draw_panel(
                annotated,
                final_text,
                current_label,
                current_conf,
                args.threshold,
                smoothed_label,
                smoothed_votes,
                get_last_word(final_text),
            )
            cv2.imshow("ASL Realtime", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                final_text = ""
            if key == ord("b") and final_text:
                final_text = final_text[:-1]
            if key == ord("s"):
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(args.save_path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {final_text}\n")

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
