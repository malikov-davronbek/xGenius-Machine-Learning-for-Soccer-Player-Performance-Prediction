import os
import sys
import pandas as pd

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import DataPreprocessor  # your class

# ===== Path to raw dataset =====
RAW_PATH = "Data/merged/df_merged.csv"
PROCESSED_PATH = "Data/processed/soccer_preprocessed.csv"

# ===== Load dataset =====
df = load_data(RAW_PATH)
print(f"📥 Raw dataset loaded | Shape: {df.shape}")

# ===== Instantiate preprocessor =====
preprocessor = DataPreprocessor(df, target_column="xG")

# ===== Run preprocessing =====
df_processed = preprocessor.process()
print(f"⚡ Preprocessing completed | Shape: {df_processed.shape}")

# ===== Optional: drop target if needed =====
# df_features_only = preprocessor.drop_target()

# ===== Save processed dataset =====
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df_processed.to_csv(PROCESSED_PATH, index=False)
print(f"✅ Preprocessed dataset saved to {PROCESSED_PATH}")
