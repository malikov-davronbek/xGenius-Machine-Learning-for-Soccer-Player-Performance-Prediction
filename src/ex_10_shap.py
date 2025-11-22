# src/shap_analysis.py
import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
from logger_setup import Logger

class ShapAnalyzer:
    def __init__(self, model, X, save_dir=None, log_file=None):
        """
        SHAP analysis for a trained model with logging.

        Parameters:
        - model: trained ML model
        - X: feature dataset (DataFrame)
        - save_dir: directory to save SHAP values and plots
        - log_file: path to log file
        """
        self.model = model
        self.X = X
        self.save_dir = save_dir

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Logger
        if log_file:
            self.logger = Logger(log_file)
            self.logger.info("Logger initialized for SHAP analysis.")
        else:
            self.logger = None

    def compute_shap_values(self):
        """Compute SHAP values for the model"""
        if self.logger:
            self.logger.info("Starting SHAP value computation...")
        explainer = shap.Explainer(self.model, self.X)
        self.shap_values = explainer(self.X)
        if self.logger:
            self.logger.info("SHAP values computed successfully.")
        return self.shap_values

    def summary_plot(self, max_display=20, show=True):
        """Generate and save summary plot"""
        plt.figure()
        shap.summary_plot(self.shap_values, self.X, max_display=max_display, show=show)
        if self.save_dir:
            plot_path = os.path.join(self.save_dir, "shap_summary_plot.png")
            plt.savefig(plot_path, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"SHAP summary plot saved to {plot_path}")
        plt.close()

    def feature_importance_table(self):
        """Return and save mean absolute SHAP values as a table"""
        shap_abs = pd.DataFrame({
            "feature": self.X.columns,
            "mean_abs_shap": abs(self.shap_values.values).mean(axis=0)
        }).sort_values(by="mean_abs_shap", ascending=False)
        if self.save_dir:
            table_path = os.path.join(self.save_dir, "shap_feature_importance.csv")
            shap_abs.to_csv(table_path, index=False)
            if self.logger:
                self.logger.info(f"SHAP feature importance table saved to {table_path}")
        return shap_abs
