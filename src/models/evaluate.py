from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a regression model on test data.

    Args:
        model: Trained regression model.
        X_test: Test features.
        y_test: Test target values.
    """
    preds = model.predict(X_test)


    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("✅ Model Evaluation Metrics:")
    print(f"R² Score      : {r2:.4f}")

    print(f"MAE           : {mae:.4f}")
    
    return {"R2": r2, "MAE": mae}
