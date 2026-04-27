import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ASL classifier from landmark CSV data.")
    parser.add_argument("--csv", default="data/landmarks.csv", help="Path to landmark CSV")
    parser.add_argument("--model-out", default="models/asl_model.joblib", help="Path to save model")
    parser.add_argument("--labels-out", default="models/labels.txt", help="Path to save labels")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    x = df.drop(columns=["label"])
    y = df["label"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    max_iter=500,
                    random_state=args.random_state,
                    early_stopping=True,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    print(f"Validation accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)

    classes = sorted(str(label) for label in model.classes_)
    os.makedirs(os.path.dirname(args.labels_out), exist_ok=True)
    with open(args.labels_out, "w", encoding="utf-8") as f:
        f.write("\n".join(classes) + "\n")

    print(f"Saved model: {args.model_out}")
    print(f"Saved labels: {args.labels_out}")


if __name__ == "__main__":
    main()
