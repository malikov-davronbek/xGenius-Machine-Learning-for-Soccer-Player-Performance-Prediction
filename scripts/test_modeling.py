# scripts/test_tuning.py
import os
import sys
import pandas as pd

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tune import tune_model  

# ===== Load engineered soccer dataset =====
DATA_PATH = "Data/engineered/soccer_features.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"📥 Dataset loaded | Shape: {df.shape}")

# ===== Prepare features & target =====
TARGET_COL = "xG"
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ===== Run GradientBoostingRegressor tuning with Optuna =====
print("⚙️ Starting GradientBoostingRegressor tuning with Optuna...")
best_params, best_value = tune_model(X, y, n_trials=10)  # Fewer trials for quick testing

print("✅ Tuning complete.")
print("Best hyperparameters:")
print(best_params)
print("Best mean cross-val score (Neg RMSE):")
print(best_value)
