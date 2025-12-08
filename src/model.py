from sklearn.ensemble import IsolationForest, RandomForestRegressor
import pandas as pd
import pickle

class CarFraudDetector:
    def __init__(self, contamination=0.01, n_estimators=100, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_names = []
        
    def train(self, X, feature_names=None):
        """
        Trains the Isolation Forest model.
        """
        self.feature_names = feature_names
        self.model.fit(X)
        
    def predict(self, X):
        """
        Predicts anomalies.
        Returns:
            predictions: -1 for outlier, 1 for inlier
            scores: anomaly scores (lower is more anomalous)
        """
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        return predictions, scores
    
    def get_anomaly_threshold(self):
        return self.model.offset_

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
