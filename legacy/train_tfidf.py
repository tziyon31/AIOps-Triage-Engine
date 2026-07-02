from pathlib import Path

import joblib  # type: ignore[import-not-found]

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # type: ignore[import-not-found]
from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]

from src.log_triage.data import load_raw_logs
from src.log_triage.pipeline import KNOWN_ACTIONS, parse_log_line

_LEGACY_DIR = Path(__file__).resolve().parent


def build_text_dataset(raw_logs):
    messages = []
    labels = []
    skipped = []

    for line in raw_logs:
        try:
            event = parse_log_line(line)
            action = event.get("action")

            if action is None:
                raise ValueError("missing label: action")

            if action not in KNOWN_ACTIONS:
                raise ValueError(f"unknown action: {action!r}")

            messages.append(event["message"])
            labels.append(action)

        except ValueError as error:
            skipped.append({
                "line": line,
                "error": str(error),
            })

    return messages, labels, skipped


def main():
    raw_logs = load_raw_logs()
    messages, labels, skipped = build_text_dataset(raw_logs)

    print("Dataset size:", len(messages))
    print("Skipped examples:", len(skipped))

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.4,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    print("\n--- TF-IDF Info ---")
    print("vocabulary size:", len(vectorizer.get_feature_names_out()))
    print("X_train_tfidf shape:", X_train_tfidf.shape)
    print("X_test_tfidf shape:", X_test_tfidf.shape)

    print("\n--- Predictions ---")
    for text, predicted, actual in zip(X_test_text, predictions, y_test):
        print("text:", text)
        print("predicted:", predicted)
        print("actual:   ", actual)
        print("---")

    accuracy = accuracy_score(y_test, predictions)

    print("\n--- Evaluation ---")
    print("accuracy:", accuracy)

    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, predictions, labels=KNOWN_ACTIONS))

    print("\nLabel order:")
    for action in KNOWN_ACTIONS:
        print(action)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, labels=KNOWN_ACTIONS))

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "known_actions": KNOWN_ACTIONS,
        "model_type": "LogisticRegression",
        "text_representation": "tfidf",
        "version": "tfidf_v1",
        "metrics": {
            "accuracy": accuracy,
        },
    }

    out_path = _LEGACY_DIR / "model_tfidf.pkl"
    joblib.dump(artifact, out_path)
    print(f"\nTF-IDF artifact saved to {out_path}")


if __name__ == "__main__":
    main()
