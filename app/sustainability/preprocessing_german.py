import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_german(path):

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(path)

    # -----------------------------
    # Check target column
    # -----------------------------
    if "target" not in df.columns:
        raise ValueError("Target column 'target' missing")

    # -----------------------------
    # Split features and labels
    # -----------------------------
    X = df.drop(columns=["target"])
    y = df["target"]

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # Scaling (numeric features already encoded)
    # -----------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
