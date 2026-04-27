import argparse
import csv
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.asl_core.hand_landmarks import HandLandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASL hand landmark data from webcam.")
    parser.add_argument("--labels", required=True, help="Comma-separated labels, e.g. A,B,C,SPACE,DEL")
    parser.add_argument("--csv", default="data/landmarks.csv", help="Path to output CSV")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if not labels:
        raise ValueError("At least one label is required.")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    new_file = not os.path.exists(args.csv)

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    current_idx = 0
    counts = {label: 0 for label in labels}

    with open(args.csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([f"f{i}" for i in range(63)] + ["label"])

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                features, annotated = extractor.extract(frame, draw=True)
                label = labels[current_idx]

                cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 120), (20, 20, 20), -1)
                cv2.putText(annotated, "Data Collection (c=capture, n/p=label, q=quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
                cv2.putText(annotated, f"Current label: {label}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(annotated, f"Captured: {counts[label]}", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 120), 2)
                cv2.imshow("ASL Data Collector", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("n"):
                    current_idx = (current_idx + 1) % len(labels)
                if key == ord("p"):
                    current_idx = (current_idx - 1) % len(labels)
                if key == ord("c") and features is not None:
                    writer.writerow(list(features) + [label])
                    counts[label] += 1

        finally:
            cap.release()
            extractor.close()
            cv2.destroyAllWindows()

    print("Saved samples by label:")
    for label in labels:
        print(f"- {label}: {counts[label]}")
    print(f"CSV path: {args.csv}")


if __name__ == "__main__":
    main()
