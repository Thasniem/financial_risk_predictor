import joblib
from sklearn.ensemble import RandomForestClassifier
from src.config import CONFIG
from src.preprocess import load_data, preprocess_data


def train_model():
    df = load_data(CONFIG["train_path"])
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = RandomForestClassifier(random_state=CONFIG["random_state"])
    model.fit(X_train, y_train)

    joblib.dump(model, CONFIG["model_path"])
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()