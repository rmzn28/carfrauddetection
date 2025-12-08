import argparse
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, clean_data
from preprocessing import preprocess_features, normalize_features
from model import CarFraudDetector, add_predictions_to_df

def main():
    parser = argparse.ArgumentParser(description="Car Fraud Detection System using Isolation Forest")
    parser.add_argument('--input', type=str, default='data/dummy_cars.csv', help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to save results')
    parser.add_argument('--contamination', type=float, default=0.05, help='Expected proportion of anomalies')
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    if df is None:
        print("Failed to load data. Exiting.")
        return
        
    # 2. Clean Data
    try:
        df_clean = clean_data(df)
    except Exception as e:
        print(f"Data cleaning error: {e}")
        return

    print(f"Data shape after cleaning: {df_clean.shape}")
    
    # 3. Preprocess
    X, df_mapped = preprocess_features(df_clean)
    X_scaled, _ = normalize_features(X)
    
    # 4. Model Training & Prediction
    detector = CarFraudDetector(contamination=args.contamination)
    detector.train(X_scaled)
    preds, scores = detector.predict(X_scaled)
    
    # 5. Results
    results = add_predictions_to_df(df_mapped, preds, scores)
    
    # Save results
    results.to_csv(args.output, index=False)
    
    # Summary
    num_anomalies = results['Is_Potential_Fraud'].sum()
    print(f"\nAnalysis Complete.")
    print(f"Total processed: {len(results)}")
    print(f"Potential Anomalies Detected: {num_anomalies}")
    print(f"Results saved to: {args.output}")
    
    if num_anomalies > 0:
        print("\nTop 5 Suspicious Listings (Most Anomalous):")
        # Sort by score (lower score = more anomalous)
        suspicious = results[results['Is_Potential_Fraud']].sort_values('Anomaly_Score').head(5)
        print(suspicious[['Brand', 'Model', 'Year', 'Price', 'Mileage', 'Anomaly_Score']])

if __name__ == "__main__":
    main()
