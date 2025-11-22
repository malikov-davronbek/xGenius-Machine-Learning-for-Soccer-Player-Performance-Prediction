# src/ex_08_improvement.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from logger_setup import Logger


class ImprovementExperiment:
    def __init__(self, df, target, test_size=0.2, random_state=42, save_dir=None, log_file=None):
        self.df = df.copy()
        self.target = target
        self.X = df.drop(columns=[target])
        self.y = df[target]
        self.test_size = test_size
        self.random_state = random_state
        self.save_dir = save_dir

        # Initialize Logger
        self.logger = Logger(log_file) if log_file else None
        if self.logger:
            self.logger.info("ImprovementExperiment initialized.")

        # Models for improvement
        self.models = {
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(alpha=0.1),
            "Ridge Regression": Ridge(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(max_depth=5),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
            "Voting Regressor": VotingRegressor(estimators=[
                ('lr', LinearRegression()),
                ('ridge', Ridge(alpha=1.0))
            ]),
            "Stacking Regressor": StackingRegressor(
                estimators=[
                    ('dt', DecisionTreeRegressor(max_depth=5)),
                    ('rf', RandomForestRegressor(n_estimators=100, max_depth=5))
                ],
                final_estimator=LinearRegression()
            )
        }

    # 1️⃣ Remove highly correlated features
    def remove_correlated_features(self, corr_threshold=0.85):
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        self.X = self.X.drop(columns=drop_cols)
        if self.logger:
            self.logger.info(f"Dropped correlated features: {drop_cols}")
        return drop_cols

    # 2️⃣ Remove irrelevant features (optionally provide list)
    def remove_irrelevant_features(self, cols_to_remove=None):
        if cols_to_remove:
            self.X = self.X.drop(columns=cols_to_remove)
            if self.logger:
                self.logger.info(f"Dropped irrelevant features: {cols_to_remove}")
        return cols_to_remove

    # 3️⃣ Train-test split
    def split_data(self):
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    # 4️⃣ Run all models
    def run_models(self):
        X_train, X_test, y_train, y_test = self.split_data()
        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            results[name] = {
                "Train MSE": mean_squared_error(y_train, y_train_pred),
                "Test MSE": mean_squared_error(y_test, y_test_pred),
                "Train R2": r2_score(y_train, y_train_pred),
                "Test R2": r2_score(y_test, y_test_pred)
            }

            if self.logger:
                self.logger.info(
                    f"{name} - Train R2: {results[name]['Train R2']:.4f}, Test R2: {results[name]['Test R2']:.4f}"
                )

        results_df = pd.DataFrame(results).T

        # Save results automatically
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            file_path = os.path.join(self.save_dir, "improvement_experiment_results.csv")
            results_df.to_csv(file_path)
            if self.logger:
                self.logger.info(f"Results saved to: {file_path}")
            print(f"✅ Results saved to: {file_path}")

        return results_df
