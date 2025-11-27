# src/models/train_pipeline.py
import os
import sys
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.utils.validate_data import simple_validate_soccer_data
from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer
from src.models.tune import tune_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load data
df = load_data("Data/merged/df_merged.csv")

# Validate
success, issues = simple_validate_soccer_data(df)
if not success:
    raise ValueError(f"Validation failed: {issues}")

# Preprocess
pre = DataPreprocessor(df, target_column="xG")
df_preprocessed = pre.process()

# Feature engineering
fe = FeatureEngineer(df_preprocessed)
df_features = fe.process()

# Prepare X, y (drop xG-derived features)
X = df_features.drop(columns=["id","player_name","xG","xGdiff","xGg"], errors="ignore")
y = df_features["xG"]

# Tune model
best_params, best_score = tune_model(X, y, n_trials=3)

# Train final pipeline
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("regressor", GradientBoostingRegressor(**best_params))
])
final_model.fit(X, y)

# Save model
model_path = "models/xG_model.joblib"
joblib.dump(final_model, model_path)
print(f"Final model saved at {model_path}")
