from pathlib import Path

import joblib  # type: ignore[import-not-found]
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
from sklearn.tree import DecisionTreeClassifier  # type: ignore[import-not-found]

from src.log_triage.data import load_raw_logs
from src.log_triage.pipeline import (
    FEATURE_NAMES,
    KNOWN_ACTIONS,
    build_dataset,
    explain_features,
)

_LEGACY_DIR = Path(__file__).resolve().parent


def decode_label(encoded_label):
    return KNOWN_ACTIONS[int(encoded_label)]


def main():
    raw_logs = load_raw_logs()
    X, y, skipped = build_dataset(raw_logs)

    print("Dataset size:", len(X))
    print("Skipped examples:", len(skipped))

    if skipped:
        print("\nSkipped:")
        for item in skipped:
            print(item)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n--- Predictions ---")
    print("predictions:", predictions.tolist())
    print("actual:", y_test)

    decoded_predictions = [decode_label(prediction) for prediction in predictions]
    decoded_actual = [decode_label(label) for label in y_test]

    print("\n--- Decoded Predictions ---")
    print("predicted:", decoded_predictions)
    print("actual:   ", decoded_actual)

    accuracy = accuracy_score(y_test, predictions)

    print("\n--- Evaluation ---")
    print("accuracy:", accuracy)

    print("\n--- Mistakes ---")
    mistake_count = 0

    for features, predicted, actual in zip(X_test, predictions, y_test):
        predicted_label = decode_label(predicted)
        actual_label = decode_label(actual)

        if predicted_label != actual_label:
            mistake_count += 1

            print(f"\nMistake #{mistake_count}")
            print("features:", explain_features(features))
            print("predicted:", predicted_label)
            print("actual:   ", actual_label)

    if mistake_count == 0:
        print("No mistakes found on this test set.")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(KNOWN_ACTIONS))),
    )

    print("\n--- Confusion Matrix ---")
    print(matrix)

    print("\nLabel order:")
    for index, action in enumerate(KNOWN_ACTIONS):
        print(index, "->", action)

    print("\nFeature count:", len(FEATURE_NAMES))

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "known_actions": KNOWN_ACTIONS,
        "model_type": "DecisionTreeClassifier",
        "version": "v1",
    }

    out_path = _LEGACY_DIR / "model.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nModel artifact saved to {out_path}")


if __name__ == "__main__":
    main()
