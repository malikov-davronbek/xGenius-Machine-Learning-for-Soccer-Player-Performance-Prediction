# src/data/preprocess.py
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

class DataPreprocessor:
    def __init__(self, df, target_column=None, logger=None):
        self.df = df.copy()
        self.target_column = target_column
        self.logger = logger

    # Handle missing values
    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].dtype in ["number"]:
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)
            else:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.df[col].fillna(mode_value, inplace=True)
        return self.df

    # Ordinal encode categorical columns
    def encode_categorical(self):
        categorical_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()
        if categorical_cols:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.df[categorical_cols] = encoder.fit_transform(self.df[categorical_cols])
        return self.df

    # Scale numeric columns
    def scale_numeric(self):
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if numeric_cols:
            scaler = MinMaxScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df

    # Full preprocessing
    def process(self):
        self.handle_missing_values()
        self.encode_categorical()
        self.scale_numeric()
        return self.df
