# 🚗 Sentetik (Second-Hand) Car Fraud Detection & Price Estimator

A comprehensive Streamlit web application powered by Machine Learning to detect potentially fraudulent car listings and estimate fair market prices.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview) 
*(Note: Replace with an actual screenshot)*

## 🌟 Features

### 🔍 Fraud Detection (Anomaly Detection)
- **Algorithm:** Uses **Random Forest Regression** to identify listings that deviate significantly from market norms.
- **Features Analyzed:** Price, Mileage, Model Year, Brand, Trim, and Body Type.
- **Smart Logic:** Flags cars that are "Too Good To Be True" (e.g., very low price for low mileage) or suspicious outliers.

### 💰 Fair Price Calculator
- **Algorithm:** Uses **Random Forest Regression** trained *only* on legitimate listing data.
- **Dynamic UI:** 
  - Intelligent filtering ensures you can't select a year that doesn't exist for a specific car model.
  - Interactive sliders and inputs for precise estimation.
- **Accuracy:** Estimates the true market value to help buyers determine if a deal is fair.

### 📊 Market Analysis Dashboard
- **Rich Visuals:** Cyberpunk-themed interactive charts (Plotly).
- **Deep Dive Stats:**
  - Price trends over years.
  - Anomaly distribution across Brands and Models.
  - Analysis of State, Condition, and Transmission types.
- **Transparency:** View top suspicious listings with detailed "Anomaly Scores".

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/car-fraud-detection.git
   cd car-fraud-detection
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Run the Streamlit App:**
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Open in Browser:**
   The app usually launches automatically at `http://localhost:8501`.

3. **Explore:**
   - Use the **Dashboard** to explore market data.
   - Switch to the **Fair Price Calculator** tab to appraise a specific vehicle.

## 📂 Project Structure

```
├── app_streamlit.py   # Main Application Entry Point
├── requirements.txt   # Dependencies
├── data/
│   └── car_prices.csv # Dataset (Place your CSV here)
├── src/
│   ├── data_loader.py    # Data Ingestion & Cleaning
│   ├── preprocessing.py  # Feature Engineering & Scaling
│   └── model.py          # ML Models (Isolation Forest & Random Forest)
└── tests/                # Verification scripts
```

## 🤖 Tech Stack
- **Python 3.8+**
- **Streamlit** (UI Framework)
- **Scikit-Learn** (Machine Learning)
- **Plotly Express** (Visualizations)
- **Pandas & NumPy** (Data Processing)

## 📄 License
MIT
