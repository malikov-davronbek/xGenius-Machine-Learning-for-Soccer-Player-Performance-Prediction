import os
import sys
import pandas as pd

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data

# ===== Path to raw dataset =====
RAW_PATH = "Data/merged/df_merged.csv"

# ===== Load the dataset =====
try:
    df = load_data(RAW_PATH)
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    print(df.head())  # optional: preview first few rows
except FileNotFoundError as e:
    print(f"❌ {e}")

# ===== Optional: save a copy or check columns =====
print(f"Columns: {df.columns.tolist()}")
