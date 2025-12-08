import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    try:
        # 'error_bad_lines' is deprecated in newer pandas, using 'on_bad_lines'
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip')
        except TypeError:
             # Fallback for older pandas versions
            df = pd.read_csv(filepath, error_bad_lines=False)
            
        print(f"Successfully loaded {len(df)} records from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Data cleaning with mapping for car_prices.csv
    """
    # 1. Standardize Column Names (Lower case first)
    df.columns = [c.lower() for c in df.columns]
    
    # 2. Map known columns to our standard schema
    # Standard: Price, Mileage, Year, Brand, Model, Trim, Body
    mapping = {
        'sellingprice': 'Price',
        'odometer': 'Mileage',
        'make': 'Brand',
        'year': 'Year',
        'model': 'Model',
        'trim': 'Trim',
        'body': 'Body',
        'state': 'State',
        'condition': 'Condition',
        'transmission': 'Transmission'
    }
    
    df = df.rename(columns=mapping)
    
    # 3. Filter for required columns
    required_ids = ['Price', 'Mileage', 'Year', 'Brand', 'Model']
    # Keep Trim and Body if they exist, but they are not strictly mandatory for basic logic, 
    # though we want them for better model.
    optional_ids = ['Trim', 'Body', 'State', 'Condition', 'Transmission']
    
    existing_cols = [c for c in required_ids + optional_ids if c in df.columns]
    df = df[existing_cols]
    
    # 4. Drop rows with missing essential values
    df = df.dropna(subset=[c for c in required_ids if c in df.columns])
    
    # 5. Type Conversions
    if 'Price' in df.columns:
        df = df[pd.to_numeric(df['Price'], errors='coerce').notnull()]
        df['Price'] = df['Price'].astype(float)
        
    if 'Mileage' in df.columns:
        df = df[pd.to_numeric(df['Mileage'], errors='coerce').notnull()]
        df['Mileage'] = df['Mileage'].astype(float)
        
    if 'Year' in df.columns:
         df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # remove rows with 0 or negative price/mileage (impossible for valid listings usually)
    df = df[df['Price'] > 100] 
    df = df[df['Mileage'] >= 0]
    
    return df.reset_index(drop=True)
