import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains a GradientBoostingRegressor model and logs it with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    with mlflow.start_run():
        # Train model
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Calculate metrics
       
        mae = mean_absolute_error(y_test, preds)                # MAE
        r2 = r2_score(y_test, preds)                             # R²

        # Log parameters, metrics, and model
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("max_depth", model.max_depth)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(model, "model")

        # Optional: log the training dataset
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. MAE: {mae:.4f}, R²: {r2:.4f}")
