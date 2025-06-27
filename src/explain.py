import matplotlib.pyplot as plt
import joblib
import pandas as pd
from config import CONFIG
from preprocess import load_data, preprocess_data

def explain_model_light():
    df = load_data(CONFIG["train_path"])
    X_train, _, y_train, _ = preprocess_data(df)
    model = joblib.load(CONFIG["model_path"])

    feature_names = pd.get_dummies(df.drop("Risk Profile", axis=1), drop_first=True).columns
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv("outputs/plots/feature_importance.csv", index=False)

    # Plot top 20
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"][:20][::-1], importance_df["Importance"][:20][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/plots/feature_importance_plot.png")
    print("✅ Feature importance plot & CSV saved.")

if __name__ == "__main__":
    explain_model_light()
