import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

class DataPreprocessor:
    def __init__(self, df, target_column=None, logger=None):
        self.df = df.copy()
        self.target_column = target_column
        self.logger = logger
        if self.logger:
            self.logger.info("DataPreprocessor initialized.")

    # 1️⃣ Handle missing values
    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].dtype in ["number"]:
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)
                if self.logger:
                    self.logger.info(f"Filled missing numeric values in '{col}' with mean.")
            else:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown"
                self.df[col].fillna(mode_value, inplace=True)
                if self.logger:
                    self.logger.info(f"Filled missing categorical values in '{col}' with mode.")
        return self.df

    # 2️⃣ Ordinal encode categorical columns
    def encode_categorical(self):
        categorical_cols = self.df.select_dtypes(exclude=["number"]).columns.tolist()
        if categorical_cols:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.df[categorical_cols] = encoder.fit_transform(self.df[categorical_cols])
            if self.logger:
                self.logger.info(f"Applied Ordinal Encoding to: {categorical_cols}")
        return self.df

    # 3️⃣ Scale numeric columns
    def scale_numeric(self):
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if numeric_cols:
            scaler = MinMaxScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            if self.logger:
                self.logger.info(f"Applied MinMax scaling to: {numeric_cols}")
        return self.df

    # 4️⃣ Drop target column if needed
    def drop_target(self):
        if self.target_column and self.target_column in self.df.columns:
            self.df.drop(columns=[self.target_column], inplace=True)
            if self.logger:
                self.logger.info(f"Dropped target column: {self.target_column}")
        return self.df

    # 5️⃣ Full preprocessing pipeline
    def process(self):
        if self.logger:
            self.logger.info("Starting preprocessing pipeline...")
        self.handle_missing_values()
        self.encode_categorical()
        self.scale_numeric()
        if self.logger:
            self.logger.info("Preprocessing completed.")
        return self.df
