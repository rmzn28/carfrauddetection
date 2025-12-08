import pandas as pd
import numpy as np
import random
import os

def generate_car_data(num_samples=1000, anomaly_ratio=0.05):
    """
    Generates synthetic car data with some intentional anomalies.
    """
    brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes']
    models = {
        'Toyota': ['Corolla', 'Camry'],
        'Honda': ['Civic', 'Accord'],
        'Ford': ['Focus', 'Fiesta'],
        'BMW': ['3 Series', '5 Series'],
        'Mercedes': ['C-Class', 'E-Class']
    }
    
    data = []
    
    # Generate normal data
    for _ in range(int(num_samples * (1 - anomaly_ratio))):
        brand = random.choice(brands)
        model = random.choice(models[brand])
        year = random.randint(2010, 2024)
        
        # Base price calculation (simplified logic)
        base_price = 20000 
        if brand in ['BMW', 'Mercedes']:
            base_price += 20000
            
        age = 2025 - year
        mileage = random.randint(0, 30000) * age
        
        # Price depreciation
        price = max(2000, base_price * (0.9 ** age) - (mileage * 0.05))
        price = price * random.uniform(0.9, 1.1) # Add noise
        
        data.append({
            'Brand': brand,
            'Model': model,
            'Year': year,
            'Mileage': int(mileage),
            'Price': int(price),
            'Condition': 'Normal'
        })
        
    # Generate anomalies ("Too good to be true" or suspicious)
    for _ in range(int(num_samples * anomaly_ratio)):
        anomaly_type = random.choice(['cheap_new', 'expensive_old', 'high_mileage_expensive'])
        
        brand = random.choice(brands)
        model = random.choice(models[brand])
        
        if anomaly_type == 'cheap_new':
            # Brand new car, absurdly low price (Scam?)
            year = 2024
            mileage = random.randint(0, 5000)
            price = random.randint(5000, 10000) # Should be 30k+
            
        elif anomaly_type == 'expensive_old':
            # Old car, absurdly high price (Error or scam?)
            year = 2010
            mileage = random.randint(150000, 200000)
            price = random.randint(50000, 100000) # Should be <10k
            
        elif anomaly_type == 'high_mileage_expensive':
             # High mileage but high price
            year = 2018
            mileage = 500000
            price = 30000 # Should be lower due to mileage
            
        data.append({
            'Brand': brand,
            'Model': model,
            'Year': year,
            'Mileage': int(mileage),
            'Price': int(price),
            'Condition': 'Anomaly'
        })
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    output_path = os.path.join("data", "dummy_cars.csv")
    os.makedirs("data", exist_ok=True)
    df = generate_car_data()
    df.to_csv(output_path, index=False)
    print(f"Dummy data generated at {output_path}")
    print(df.head())
