from sklearn.ensemble import IsolationForest, RandomForestRegressor
import pandas as pd
import pickle

import numpy as np

class CarFraudDetector:
    def __init__(self, threshold=0.30, n_estimators=100, random_state=42):
        """
        Anomaly Detector based on Price Prediction Residuals.
        Args:
            threshold (float): Percentage deviation to flag as anomaly (e.g., 0.30 for 30%).
                               If |(Actual - Predicted) / Predicted| > threshold, it's an anomaly.
            n_estimators (int): Trees in Random Forest.
        """
        self.threshold = threshold
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = []
        
    def train(self, X, y, feature_names=None):
        """
        Trains the Price Predictor.
        Args:
            X: Feature matrix (MUST NOT INCLUDE PRICE)
            y: Target Price
        """
        self.feature_names = feature_names
        self.model.fit(X, y)
        
    def predict(self, X, y_actual):
        """
        Predicts anomalies based on price deviation.
        Args:
            X: Feature matrix
            y_actual: Actual prices to compare against
        Returns:
            predictions: -1 for outlier/anomaly, 1 for normal
            scores: Percentage deviation (Official metric for anomaly score)
        """
        y_pred = self.model.predict(X)
        
        # Avoid division by zero
        safe_y_pred = np.maximum(y_pred, 1.0)
        
        # Calculate Deviation (Residual Percentage)
        # Positive score = Actual is higher than predicted (Expensive)
        # Negative score = Actual is lower than predicted (Cheap)
        residuals_pct = (y_actual - safe_y_pred) / safe_y_pred
        
        # Flag if absolute deviation > threshold
        is_anomaly = np.abs(residuals_pct) > self.threshold
        
        # -1 for Anomaly, 1 for Normal
        predictions = np.where(is_anomaly, -1, 1)
        
        return predictions, residuals_pct
    
    def get_anomaly_threshold(self):
        return self.threshold

class CarPricePredictor:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

def add_predictions_to_df(df, predictions, scores):
    """
    Adds results back to the original dataframe.
    """
    df_out = df.copy()
    df_out['Anomaly_Score'] = scores
    df_out['Is_Anomaly'] = predictions
    # Convert predictions: -1 (Anomaly) -> True, 1 (Normal) -> False
    df_out['Is_Potential_Fraud'] = df_out['Is_Anomaly'] == -1
    
    return df_out
