import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(train_path):
    return pd.read_csv(train_path)

def preprocess_data(df):
    df = df.copy()
    # Drop rows with missing target
    df = df.dropna(subset=["Risk Profile"])

    X = df.drop("Risk Profile", axis=1)
    y = df["Risk Profile"]

    # Fill missing values
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Scale numerical values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)