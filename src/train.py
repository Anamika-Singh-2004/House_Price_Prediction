import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import DATA_PATH, MODEL_DIR, MODEL_PATH, METRICS_PATH, TARGET, FEATURES
from .preprocessing import clean_data, build_preprocessor

MODELS = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
}

def train_model(model_name="Random Forest", test_size=0.2, data_path=DATA_PATH):
    df = clean_data(pd.read_csv(data_path))
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", MODELS[model_name])
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return pipe, metrics, y_test, pred

if __name__ == "__main__":
    _, metrics, _, _ = train_model()
    print(json.dumps(metrics, indent=2))
