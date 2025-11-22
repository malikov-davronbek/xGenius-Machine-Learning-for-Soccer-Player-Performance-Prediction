# src/ex_07_experiment.py

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os

class Experiment:
    """
    Baseline Experiment Class for Soccer Player Performance Modeling.
    Uses simple algorithms: Linear Regression, Decision Tree, KNN, SVM.
    Includes train/test evaluation for checking overfitting/underfitting.
    """

    def __init__(self, df, target, test_size=0.2, random_state=42, save_dir=None):
        self.df = df.copy()
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.test_size = test_size
        self.random_state = random_state
        self.save_dir = save_dir

        # Initialize only baseline models
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=5),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "SVM": SVR(C=1.0, epsilon=0.1)
        }

    def run(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        results = {}
        for name, model in self.models.items():
            # Train
            model.fit(X_train, y_train)
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            results[name] = {
                "Train_MSE": mean_squared_error(y_train, y_train_pred),
                "Train_R2": r2_score(y_train, y_train_pred),
                "Test_MSE": mean_squared_error(y_test, y_test_pred),
                "Test_R2": r2_score(y_test, y_test_pred)
            }

        # Convert to DataFrame
        results_df = pd.DataFrame(results).T

        # Save automatically if save_dir is provided
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            file_path = os.path.join(self.save_dir, "baseline_experiment_results.csv")
            results_df.to_csv(file_path)
            print(f"✅ Baseline results saved to: {file_path}")

        return results_df
