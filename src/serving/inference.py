# src/serving/inference_pipeline.py
import os
import sys
import pandas as pd
import joblib
import numpy as np

# Add project root for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.utils.validate_data import simple_validate_soccer_data
from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer

class InferencePipeline:
    """
    Accepts raw player stats, computes all engineered features,
    fills missing columns, and predicts using the trained model.
    """
    def __init__(self, model_path: str, features_path: str, logger=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        self.logger = logger

    def predict(self, input_dict: dict):
        # Convert raw input to DataFrame
        df = pd.DataFrame([input_dict])

        # 1️⃣ Validate input
        is_valid, errors = simple_validate_soccer_data(df, require_target=False)
        if not is_valid:
            return {"success": False, "errors": errors}

        # 2️⃣ Preprocess
        df = DataPreprocessor(df=df, target_column=None, logger=self.logger).process()

        # 3️⃣ Feature engineering
        fe = FeatureEngineer(df=df, logger=self.logger)
        df = fe.process(include_xg_features=True)

        # 4️⃣ Drop irrelevant columns
        drop_cols = ["id", "player_name"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # 5️⃣ Fill missing columns with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # 6️⃣ Reorder columns to match training
        df = df[self.feature_names]

        # 7️⃣ Predict
        try:
            prediction = float(self.model.predict(df)[0])
            return {"success": True, "prediction": prediction}
        except Exception as e:
            return {"success": False, "errors": [str(e)]}
