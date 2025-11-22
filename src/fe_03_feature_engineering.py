import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df, logger=None):
        self.df = df.copy()
        self.logger = logger
        if self.logger:
            self.logger.info("FeatureEngineer initialized.")

    # Helper: safe division using np.where
    def safe_divide(self, numerator, denominator):
        return np.where(denominator != 0, numerator / denominator, np.nan)

    # 1️⃣ Goals per game & per minute
    def goals_features(self):
        self.df['aGg'] = self.safe_divide(self.df['goals'], self.df['games'])
        self.df['gpm'] = self.safe_divide(self.df['goals'], self.df['time'])
        if self.logger:
            self.logger.info("Created goal-related features: aGg, gpm")
        return self.df

    # 2️⃣ Assists per game & per minute
    def assists_features(self):
        self.df['apg'] = self.safe_divide(self.df['assists'], self.df['games'])
        self.df['apm'] = self.safe_divide(self.df['assists'], self.df['time'])
        if self.logger:
            self.logger.info("Created assist-related features: apg, apm")
        return self.df

    # 3️⃣ Shots per game & per minute
    def shots_features(self):
        self.df['shpg'] = self.safe_divide(self.df['shots'], self.df['games'])
        self.df['shpm'] = self.safe_divide(self.df['shots'], self.df['time'])
        if self.logger:
            self.logger.info("Created shot-related features: shpg, shpm")
        return self.df

    # 4️⃣ Key passes per game & per minute
    def key_passes_features(self):
        self.df['kppg'] = self.safe_divide(self.df['key_passes'], self.df['games'])
        self.df['kppm'] = self.safe_divide(self.df['key_passes'], self.df['time'])
        if self.logger:
            self.logger.info("Created key pass-related features: kppg, kppm")
        return self.df

    # 5️⃣ Cards per game & per minute
    def cards_features(self):
        self.df['ypg'] = self.safe_divide(self.df['yellow_cards'], self.df['games'])
        self.df['ypm'] = self.safe_divide(self.df['yellow_cards'], self.df['time'])
        self.df['rpg'] = self.safe_divide(self.df['red_cards'], self.df['games'])
        self.df['rpm'] = self.safe_divide(self.df['red_cards'], self.df['time'])
        if self.logger:
            self.logger.info("Created card-related features: ypg, ypm, rpg, rpm")
        return self.df

    # 6️⃣ xG differences & per game
    def xg_features(self):
        self.df['xGdiff'] = self.df['goals'] - self.df['xG']
        self.df['xGg'] = self.safe_divide(self.df['xG'], self.df['games'])
        if self.logger:
            self.logger.info("Created xG-related features: xGdiff, xGg")
        return self.df

    # 7️⃣ Full feature engineering pipeline
    def process(self):
        if self.logger:
            self.logger.info("Starting feature engineering pipeline...")
        self.goals_features()
        self.assists_features()
        self.shots_features()
        self.key_passes_features()
        self.cards_features()
        self.xg_features()
        if self.logger:
            self.logger.info("Feature engineering completed.")
        return self.df
