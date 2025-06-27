from src.train import train_model
from src.evaluate import evaluate_model
from src.explain import explain_model

if __name__ == "__main__":
    print(">> Training model...")
    train_model()
    print(">> Model training complete.")
    print(">> Evaluating model...")
    evaluate_model()
    print(">> Evaluation complete.")
    print(">> Generating SHAP explainability...")
    explain_model()
    print(">> Explainability generated.")