import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from logger_setup import Logger


class HyperparameterTuner:

    def __init__(self, X, y, save_dir_results, save_dir_models, log_file=None):
        """
        Runs hyperparameter tuning and saves:
        - Best parameters
        - Best model scores
        - Best tuned models (.joblib)
        """

        self.X = X
        self.y = y

        # Save directories
        self.save_dir_results = save_dir_results
        self.save_dir_models = save_dir_models
        os.makedirs(save_dir_results, exist_ok=True)
        os.makedirs(save_dir_models, exist_ok=True)

        # Logger
        if log_file:
            self.logger = Logger(log_file)
        else:
            self.logger = Logger(os.path.join(save_dir_results, "tuning.log"))

        # Store results
        self.best_params = {}
        self.best_scores = {}
        self.best_models = {}

        self.logger.info("HyperparameterTuner initialized successfully.")

    def _define_search_space(self):
        """Parameter grids for tuning"""
        return {
            "LinearRegression": {"model": LinearRegression(), "params": {"fit_intercept": [True, False]}},
            "Ridge": {"model": Ridge(), "params": {"alpha": [0.1, 1.0, 5.0, 10.0]}},
            "Lasso": {"model": Lasso(), "params": {"alpha": [0.001, 0.01, 0.1, 1.0]}},
            "DecisionTree": {"model": DecisionTreeRegressor(), "params": {"max_depth": [3,5,10,None], "min_samples_split":[2,5,10]}},
            "RandomForest": {"model": RandomForestRegressor(), "params": {"n_estimators":[50,100,200], "max_depth":[None,5,10]}},
            "KNN": {"model": KNeighborsRegressor(), "params": {"n_neighbors":[3,5,7,11], "weights":["uniform","distance"]}},
            "SVM": {"model": SVR(), "params": {"kernel":["rbf","linear"], "C":[0.1,1,10]}},
            "GradientBoosting": {"model": GradientBoostingRegressor(), "params": {"n_estimators":[100,200], "learning_rate":[0.01,0.05,0.1]}}
        }

    def run_tuning(self):
        """Runs GridSearchCV for all models"""
        search_space = self._define_search_space()

        for model_name, cfg in search_space.items():
            self.logger.info(f"Starting tuning for {model_name}...")
            model = cfg["model"]
            params = cfg["params"]

            gs = GridSearchCV(model, params, cv=5, scoring="r2", n_jobs=-1)
            try:
                gs.fit(self.X, self.y)

                self.best_params[model_name] = gs.best_params_
                self.best_scores[model_name] = gs.best_score_
                self.best_models[model_name] = gs.best_estimator_

                # Save model
                model_path = os.path.join(self.save_dir_models, f"{model_name}_best.joblib")
                joblib.dump(gs.best_estimator_, model_path)

                self.logger.info(f"✔ {model_name} tuned successfully.")
                self.logger.info(f"   → Best Params: {gs.best_params_}")
                self.logger.info(f"   → Best CV Score: {gs.best_score_:.4f}")

            except Exception as e:
                self.logger.error(f"❌ Error tuning {model_name}: {e}")

        self._save_results()
        self.logger.info("All model tuning completed.")

    def _save_results(self):
        """Save best params and scores to CSV"""
        df_params = pd.DataFrame.from_dict(self.best_params, orient='index')
        df_scores = pd.DataFrame.from_dict(self.best_scores, orient='index', columns=["Best_R2_Score"])
        df_final = df_params.join(df_scores)
        output_path = os.path.join(self.save_dir_results, "final_tuning_results.csv")
        df_final.to_csv(output_path)
        self.logger.info(f"Results saved to {output_path}")
