import json
from sklearn.metrics import classification_report, accuracy_score
from src.config import CONFIG
from src.train import train_model


def evaluate_model():
    model, X_test, y_test = train_model()
    y_pred = model.predict(X_test)

    metrics = classification_report(y_test, y_pred, output_dict=True)
    metrics["accuracy"] = accuracy_score(y_test, y_pred)

    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics saved.")

if __name__ == "__main__":
    evaluate_model()