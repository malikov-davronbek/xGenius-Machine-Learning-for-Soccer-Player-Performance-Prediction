# src/models/tune.py
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def tune_model(X: pd.DataFrame, y: pd.Series, n_trials: int = 20):
    """
    Tunes a GradientBoostingRegressor using Optuna and returns best hyperparameters and best score.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        n_trials (int): Number of Optuna trials.

    Returns:
        tuple: (best_params, best_value)
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42
        }

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", GradientBoostingRegressor(**params))
        ])

        scores = cross_val_score(
            model,
            X,
            y,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            error_score="raise"
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value  # This is the highest mean cross-val score
    print("Best Params:", best_params)
    print("Best Score (Neg RMSE):", best_value)
    return best_params, best_value
