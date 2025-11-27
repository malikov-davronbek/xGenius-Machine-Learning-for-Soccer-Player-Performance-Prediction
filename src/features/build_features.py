# src/features/feature_engineer.py
import os
import sys
import pandas as pd
import numpy as np

# Add project root for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

class FeatureEngineer:
    def __init__(self, df, logger=None):
        self.df = df.copy()
        self.logger = logger

    def safe_divide(self, numerator, denominator):
        return np.where(denominator != 0, numerator / denominator, np.nan)

    def goals_features(self):
        self.df['aGg'] = self.safe_divide(self.df['goals'], self.df['games'])
        self.df['gpm'] = self.safe_divide(self.df['goals'], self.df['time'])
        return self.df

    def assists_features(self):
        self.df['apg'] = self.safe_divide(self.df['assists'], self.df['games'])
        self.df['apm'] = self.safe_divide(self.df['assists'], self.df['time'])
        return self.df

    def shots_features(self):
        self.df['shpg'] = self.safe_divide(self.df['shots'], self.df['games'])
        self.df['shpm'] = self.safe_divide(self.df['shots'], self.df['time'])
        return self.df

    def key_passes_features(self):
        self.df['kppg'] = self.safe_divide(self.df['key_passes'], self.df['games'])
        self.df['kppm'] = self.safe_divide(self.df['key_passes'], self.df['time'])
        return self.df

    def cards_features(self):
        self.df['ypg'] = self.safe_divide(self.df['yellow_cards'], self.df['games'])
        self.df['ypm'] = self.safe_divide(self.df['yellow_cards'], self.df['time'])
        self.df['rpg'] = self.safe_divide(self.df['red_cards'], self.df['games'])
        self.df['rpm'] = self.safe_divide(self.df['red_cards'], self.df['time'])
        return self.df

    def xg_features(self):
        # Compute xG if missing
        if "xG" not in self.df.columns:
            if "npxG" in self.df.columns and "xGBuildup" in self.df.columns:
                self.df["xG"] = self.df["npxG"] + self.df["xGBuildup"]
            elif "npg" in self.df.columns:
                self.df["xG"] = self.df["npg"]
            else:
                self.df["xG"] = 0
        self.df['xGdiff'] = self.df['goals'] - self.df['xG']
        self.df['xGg'] = self.safe_divide(self.df['xG'], self.df['games'])
        return self.df

    def process(self, include_xg_features=True):
        self.goals_features()
        self.assists_features()
        self.shots_features()
        self.key_passes_features()
        self.cards_features()
        if include_xg_features:
            self.xg_features()
        return self.df
