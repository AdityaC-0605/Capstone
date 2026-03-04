from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_logistic(X_train, y_train, preprocessor):
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0
        ))
    ])

    model.fit(X_train, y_train)
    return model
