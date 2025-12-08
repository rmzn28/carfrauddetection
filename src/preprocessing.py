import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.categorical_cols = ['Brand', 'Model', 'Trim', 'Body']
        self.numerical_cols = ['Price', 'Mileage', 'Age']
        
    def fit_transform(self, df, exclude_cols=None):
        """
        Preprocesses the dataframe: Feature Engineering -> Encoding -> Scaling
        Args:
            df: Input dataframe
            exclude_cols: List of columns to exclude from the feature matrix X (e.g. ['Price'] for regression)
        Returns:
            X_scaled: The matrix used for the model
            df: The ORIGINAL dataframe (with added Age feature) but keeping text columns for display
            feature_cols: List of features used
        """
        if exclude_cols is None:
            exclude_cols = []

        df_display = df.copy()
        
        # 1. Feature Engineering: Model Age
        current_year = 2025
        df_display['Age'] = current_year - df_display['Year']
        
        # Create a separate copy for modeling so we don't mess up the display df
        df_model = df_display.copy()

        # 2. Categorical Encoding (Label Encoding)
        for col in self.categorical_cols:
            if col in df_model.columns and col not in exclude_cols:
                # Convert to string
                df_model[col] = df_model[col].astype(str)
                le = LabelEncoder()
                df_model[col] = le.fit_transform(df_model[col])
                self.encoders[col] = le
        
        # 3. Select Features for Model
        available_cats = [c for c in self.categorical_cols if c in df_model.columns and c not in exclude_cols]
        available_nums = [c for c in self.numerical_cols if c in df_model.columns and c not in exclude_cols]
        
        feature_cols = available_nums + available_cats
        
        X = df_model[feature_cols]
        
        # 4. Scaling
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, df_display, feature_cols

    def transform(self, df):
        """
        Applies existing transformations to new data (for single prediction) - Not strictly needed for this batch app but good practice.
        """
        # (Simplified for this task since we re-train often in the Streamlit app flow)
        pass 
