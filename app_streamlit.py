import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, clean_data
from preprocessing import DataPreprocessor
from model import CarFraudDetector, add_predictions_to_df, CarPricePredictor

# Page Config
st.set_page_config(
    page_title="Car Fraud Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Cyberpunk/Modern" feel
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4c4c4c;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 Second-Hand Car Fraud Detection")
# --- Sidebar ---
st.sidebar.header("Configuration")

# Data Loading - Cached
@st.cache_data
def get_and_clean_data(filepath):
    df = load_data(filepath)
    if df is not None:
        df_clean = clean_data(df)
        return df_clean
    return None

data_path = os.path.join("data", "car_prices.csv")

# Simplified Data Loading Logic
if os.path.exists(data_path):
    df = get_and_clean_data(data_path)
    if df is not None:
        st.sidebar.success(f"Loaded Database ({len(df):,} records)")
        if st.sidebar.button("↻ Reload Data / Clear Cache"):
            st.cache_data.clear()
            st.rerun()
else:
    st.error("Data file not found. Please ensure 'data/car_prices.csv' exists.")
    st.stop()

# Model Parameters
st.sidebar.subheader("Model Settings")
contamination = st.sidebar.slider("Anomaly Sensitivity", 0.001, 0.05, 0.01, 0.001, help="Higher = More strict (Flags more cars).")
n_estimators = 100 # Fixed to improve speed, or make optional

# --- Main App Logic ---

# Caching the Heavy Computation
@st.cache_data(show_spinner=False)
def train_and_predict(df, contamination_rate):
    # Preprocess
    preprocessor = DataPreprocessor()
    X_scaled, df_display, features = preprocessor.fit_transform(df)
    
    # Train
    model = CarFraudDetector(contamination=contamination_rate, n_estimators=100)
    model.train(X_scaled, feature_names=features)
    
    # Predict
    preds, scores = model.predict(X_scaled)
    
    # Results - Use the df_display which has original text columns!
    results = add_predictions_to_df(df_display, preds, scores)
    return results

with st.spinner('Analyzing market data... (This runs once)'):
    results = train_and_predict(df, contamination)

# --- Layout Architecture ---

main_tab1, main_tab2 = st.tabs(["📊 Market Analysis Dashboard", "💰 Fair Price Calculator"])

# ==========================================
# TAB 1: MARKET ANALYSIS (Filters + Charts)
# ==========================================
with main_tab1:
    # 1. Dashboard Filters
    all_brands = sorted(results['Brand'].unique().tolist())
    selected_brand = st.selectbox("Quick Filter by Brand", ["All"] + all_brands, key="analysis_brand_filter")

    # Apply View Filter
    if selected_brand != "All":
        view_df = results[results['Brand'] == selected_brand]
    else:
        view_df = results

    # 2. Metrics
    n_anomalies = view_df['Is_Potential_Fraud'].sum()
    avg_price_fraud = view_df[view_df['Is_Potential_Fraud']]['Price'].mean()
    avg_price_normal = view_df[~view_df['Is_Potential_Fraud']]['Price'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", f"{len(view_df):,}")
    col2.metric("Suspicious Listings", f"{n_anomalies}", delta="-Flagged" if n_anomalies > 0 else "Clean")

    if n_anomalies > 0:
        diff = avg_price_fraud - avg_price_normal
        col3.metric("Avg Fraud Price", f"${avg_price_fraud:,.0f}", f"{diff:+,.0f} vs Normal")
    else:
        col3.metric("Avg Market Price", f"${avg_price_normal:,.0f}")

    # 3. Visualizations Sub-Tabs
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["📉 Price Analysis", "🔍 Inspection Lists", "📊 Stats"])

    with sub_tab1:
        st.subheader(f"Price vs Mileage Analysis ({selected_brand})")
        
        # Sample data if too large
        MAX_POINTS = 50000
        if len(view_df) > MAX_POINTS:
            st.caption(f"⚠️ Displaying random {MAX_POINTS:,} records out of {len(view_df):,} for performance.")
            chart_df = view_df.sample(MAX_POINTS)
        else:
            chart_df = view_df

        fig = px.scatter(
            chart_df, 
            x='Mileage', 
            y='Price', 
            color='Is_Potential_Fraud',
            hover_data=['Brand', 'Model', 'Year', 'Trim'], 
            color_discrete_map={False: '#3366cc', True: '#ff2b2b'},
            opacity=0.7,
            height=500
        )
        fig.update_layout(legend_title_text='Is Suspicious?')
        st.plotly_chart(fig, use_container_width=True)

    with sub_tab2:
        st.subheader("Deep Dive")
        list_tab1, list_tab2 = st.tabs(["🔴 Suspicious Listings", "🟢 Normal Listings"])
        
        with list_tab1:
            if n_anomalies > 0:
                fraud_df = view_df[view_df['Is_Potential_Fraud']].sort_values('Anomaly_Score')
                st.dataframe(
                    fraud_df[['Brand', 'Model', 'Year', 'Trim', 'Price', 'Mileage', 'Anomaly_Score']].head(1000),
                    use_container_width=True
                )
                st.caption("Showing top 1,000 most suspicious records.")
            else:
                st.info("No anomalies detected.")
                
        with list_tab2:
            normal_df = view_df[~view_df['Is_Potential_Fraud']]
            if len(normal_df) > 0:
                 st.dataframe(
                    normal_df[['Brand', 'Model', 'Year', 'Trim', 'Price', 'Mileage']].sample(min(1000, len(normal_df))),
                    use_container_width=True
                )
                 st.caption("Showing random 1,000 normal records.")
            else:
                st.info("No normal records found (?)")

    with sub_tab3:
        st.subheader("Suspicious Listings Statistics")
        st.caption("These charts analyze ONLY the listings flagged as potential fraud.")
        
        suspicious_df = view_df[view_df['Is_Potential_Fraud']]
        
        if len(suspicious_df) == 0:
            st.warning("No suspicious listings found with current sensitivity settings.")
        else:
            # --- ROW 1: BRAND & MODEL (Bar Charts using Gradient) ---
            r1_col1, r1_col2 = st.columns(2)
            
            with r1_col1:
                st.markdown("#### Top Suspicious Brands")
                top_makes = suspicious_df['Brand'].value_counts().head(10).reset_index()
                top_makes.columns = ['Brand', 'Count']
                
                fig_make = px.bar(
                    top_makes, 
                    x='Brand', 
                    y='Count', 
                    color='Count',
                    color_continuous_scale='RdPu', # Stylish Red-Purple gradient
                    text_auto=True
                )
                fig_make.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False, template="plotly_dark")
                st.plotly_chart(fig_make, use_container_width=True)

            with r1_col2:
                st.markdown("#### Top Suspicious Models")
                top_models = suspicious_df['Model'].value_counts().head(10).reset_index()
                top_models.columns = ['Model', 'Count']
                
                fig_model = px.bar(
                    top_models, 
                    x='Model', 
                    y='Count', 
                    color='Count',
                    color_continuous_scale='Viridis', # Stylish Green-Blue gradient
                    text_auto=True
                )
                fig_model.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False, template="plotly_dark")
                st.plotly_chart(fig_model, use_container_width=True)
            
            # --- ROW 2: YEAR TREND (Area Chart - Fixed Missing Years) ---
            st.markdown("#### Suspicious Year Trend")
            
            # Logic to fill missing years
            year_counts = suspicious_df['Year'].value_counts().reset_index()
            year_counts.columns = ['Year', 'Count']
            
            if not year_counts.empty:
                min_yr, max_yr = int(year_counts['Year'].min()), int(year_counts['Year'].max())
                full_range = pd.DataFrame({'Year': range(min_yr, max_yr + 1)})
                year_data = full_range.merge(year_counts, on='Year', how='left').fillna(0)
                
                fig_year = px.area(
                    year_data, 
                    x='Year', 
                    y='Count', 
                    markers=True,
                    color_discrete_sequence=['#ff4b4b'] # Streamlit Red
                )
                fig_year.update_layout(xaxis_title=None, yaxis_title="Number of Listings", template="plotly_dark")
                st.plotly_chart(fig_year, use_container_width=True)
            else:
                st.info("Not enough year data to plot trend.")

            # --- ROW 3: STATE, CONDITION, TRANSMISSION ---
            r3_col1, r3_col2, r3_col3 = st.columns(3)

            with r3_col1:
                st.markdown("#### Top States")
                if 'State' in suspicious_df.columns:
                    top_states = suspicious_df['State'].value_counts().head(10).reset_index()
                    top_states.columns = ['State', 'Count']
                    # Horizontal Bar for states is usually better
                    fig_state = px.bar(
                        top_states, 
                        x='Count', 
                        y='State', 
                        orientation='h', 
                        color='Count', 
                        color_continuous_scale='Oranges'
                    )
                    fig_state.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title=None, yaxis_title=None, coloraxis_showscale=False, template="plotly_dark")
                    st.plotly_chart(fig_state, use_container_width=True)
                else:
                    st.error("State data missing.")

            with r3_col2:
                st.markdown("#### Condition")
                if 'Condition' in suspicious_df.columns:
                    fig_cond = px.histogram(
                        suspicious_df, 
                        x='Condition', 
                        nbins=20,
                        color_discrete_sequence=['#00CC96']
                    )
                    fig_cond.update_layout(xaxis_title="Condition Score", yaxis_title=None, template="plotly_dark", bargap=0.1)
                    st.plotly_chart(fig_cond, use_container_width=True)
                else:
                    st.error("Condition data missing.")

            with r3_col3:
                st.markdown("#### Transmission")
                if 'Transmission' in suspicious_df.columns:
                    top_trans = suspicious_df['Transmission'].value_counts().reset_index()
                    top_trans.columns = ['Transmission', 'Count']
                    fig_trans = px.pie(
                        top_trans, 
                        values='Count', 
                        names='Transmission', 
                        hole=0.5,
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig_trans.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_trans, use_container_width=True)
                else:
                    st.error("Transmission data missing.")

# ==========================================
# TAB 2: FAIR PRICE CALCULATOR
# ==========================================
with main_tab2:
    st.subheader("💰 Fair Price Calculator")
    st.markdown("Estimate the market price for a car based on current **Verified Normal** listings.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    # 1. Brand Selection
    p_brands = sorted(df['Brand'].unique().tolist())
    p_brand = col_p1.selectbox("Select Brand", p_brands, key="calc_brand")
    
    # 2. Model Selection (Dynamic)
    filtered_models = df[df['Brand'] == p_brand]
    p_models = sorted(filtered_models['Model'].unique().tolist())
    p_model = col_p2.selectbox("Select Model", p_models, key="calc_model")
    
    # 3. Trim Selection (Dynamic)
    filtered_trims = filtered_models[filtered_models['Model'] == p_model]
    if 'Trim' in df.columns:
        # Filter trims that actually exist for this model
        valid_trims = filtered_trims['Trim'].astype(str).unique().tolist()
        p_trims = sorted([t for t in valid_trims if t.lower() != 'nan'])
        if not p_trims: p_trims = ["Standard"]
        p_trim = col_p3.selectbox("Select Trim", p_trims, key="calc_trim")
    else:
        p_trim = "Unknown"

    col_p4, col_p5 = st.columns(2)
    
    # 4. Year Selection (Dynamic Check)
    # user asked to restrict years to available ones
    available_years = sorted(filtered_trims['Year'].unique().tolist())
    if not available_years: # Fallback
        min_y, max_y = 2000, 2025
        available_years = range(2000, 2026)
    
    # Use selectbox for strict adherence, or slider with bounds
    # Selectbox is safer against "1986" if it doesnt exist
    p_year = col_p4.selectbox("Select Year", available_years, index=len(available_years)-1 if available_years else 0, help="Only displaying years available in the dataset for this model.")
    
    # UI Change: Use number input for consistent look
    p_mileage = col_p5.number_input("Mileage", min_value=0, max_value=500000, value=50000, step=1000, key="calc_mileage")
    
    if st.button("Calculate Fair Price", type="primary"):
        # Train on clean data only
        clean_data_for_training = results[~results['Is_Potential_Fraud']]
        
        # 1. Train Regressor on the fly
        from model import CarPricePredictor
        
        with st.spinner("Analyzing verified market data..."):
            input_data = {
                'Brand': [p_brand],
                'Model': [p_model],
                'Trim': [p_trim], 
                'Year': [p_year],
                'Mileage': [p_mileage],
                'Price': [0], # Dummy
                'Body': ['Unknown']
            }
            if 'Body' in df.columns:
                 common_body = filtered_trims['Body'].mode()
                 if not common_body.empty:
                     input_data['Body'] = [common_body[0]]
            
            # --- Logic Restore Start ---
            input_df = pd.DataFrame(input_data)
            
            # Combine & Preprocess
            train_df = clean_data_for_training.copy()
            combined_df = pd.concat([train_df, input_df], axis=0)
            
            # Preprocess - EXCLUDE PRICE from features for Regression!
            preprocessor_reg = DataPreprocessor()
            # We want to predict Price, so it shouldn't be in X
            X_combined_scaled, _, _ = preprocessor_reg.fit_transform(combined_df, exclude_cols=['Price'])
            
            # Split back
            X_train = X_combined_scaled[:-1]
            y_train = train_df['Price'].values
            X_test = X_combined_scaled[-1:]
            
            predictor = CarPricePredictor(n_estimators=50) # Faster training for UI
            # --- Logic Restore End ---

            predictor.train(X_train, y_train)
            predicted_price = predictor.predict(X_test)[0]
            
            st.success(f"Estimated Fair Price: **${predicted_price:,.0f}**")
            st.caption(f"Based on verified normal listings.")
