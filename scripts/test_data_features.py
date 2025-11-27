import os
import sys
import pandas as pd

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.features.build_features import FeatureEngineer  # your class

# ===== Paths =====
PROCESSED_PATH = "Data/processed/soccer_preprocessed.csv"  # output from preprocessing
FEATURES_PATH = "Data/engineered/soccer_features.csv"      # final feature dataset

# ===== Load preprocessed dataset =====
df = load_data(PROCESSED_PATH)
print(f"📥 Preprocessed dataset loaded | Shape: {df.shape}")

# ===== Instantiate FeatureEngineer =====
engineer = FeatureEngineer(df)

# ===== Run feature engineering pipeline =====
df_features = engineer.process()
print(f"⚡ Feature engineering completed | Shape: {df_features.shape}")

# ===== Save final dataset =====
os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
df_features.to_csv(FEATURES_PATH, index=False)
print(f"✅ Features dataset saved to {FEATURES_PATH}")
