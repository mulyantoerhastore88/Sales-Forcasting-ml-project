import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Sales Forecast AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helper Functions ----------
@st.cache_resource
def load_model():
    """Load the trained model from file."""
    # Get the absolute path to the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '..', 'Model', 'sales_model.pkl')
    
    # Alternative: if model is in same directory as app.py
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'sales_model.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("🚨 Model file not found. Please ensure 'sales_model.pkl' is in the 'Model' folder.")
        st.info("If you haven't uploaded the model, please add it to your repository or contact the developer.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(base_dir, '..', 'data', 'sample_sales_data.csv')
    
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        # Create dummy data if sample not available
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        data = {
            'Store_ID': np.random.choice([101, 102, 103], size=100),
            'Date': dates,
            'Sales': np.random.randint(200, 800, size=100),
            'Promotion': np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
            'Holiday': np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
            'Price': np.random.uniform(10, 50, size=100).round(2)
        }
        return pd.DataFrame(data)

def predict_sales(model, features):
    """Make prediction using the loaded model."""
    # features should be a DataFrame with correct columns
    prediction = model.predict(features)[0]
    return round(prediction, 2)

# ---------- Load Model ----------
model = load_model()

# ---------- Sidebar Inputs ----------
st.sidebar.title("📋 Input Parameters")
st.sidebar.markdown("Adjust the values to see predicted sales.")

store_id = st.sidebar.selectbox("Store ID", [101, 102, 103, 104, 105])
promotion = st.sidebar.selectbox("Promotion Active?", ["Yes", "No"])
holiday = st.sidebar.selectbox("Is it a Holiday?", ["Yes", "No"])
price = st.sidebar.number_input("Product Price ($)", min_value=5.0, max_value=200.0, value=25.0, step=0.5)
day_of_week = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
lag_sales = st.sidebar.number_input("Previous Day Sales (units)", min_value=0, max_value=2000, value=300)

# Convert categorical to numerical
promo_binary = 1 if promotion == "Yes" else 0
holiday_binary = 1 if holiday == "Yes" else 0
day_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}
day_num = day_map[day_of_week]

# Create feature dataframe (order and names must match training)
# Assuming your model was trained with these feature names:
feature_names = ['Store_ID', 'Promotion', 'Holiday', 'Price', 'Day_of_Week', 'Lag_Sales']
input_features = pd.DataFrame([[store_id, promo_binary, holiday_binary, price, day_num, lag_sales]],
                              columns=feature_names)

# ---------- Main Page ----------
st.title("📈 Sales Forecasting AI")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Prediction")
    if model is not None:
        try:
            prediction = predict_sales(model, input_features)
            st.metric("Predicted Sales (units)", f"{prediction:,.0f}")
            
            # Show a simple gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Forecast"},
                gauge = {
                    'axis': {'range': [0, 1000]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 300], 'color': "lightgray"},
                        {'range': [300, 600], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 800
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.warning("Model not available. Please check the deployment.")

with col2:
    st.subheader("📊 Feature Importance")
    st.image("Images/feature_importances.png", use_column_width=True)
    st.caption("Top drivers: Promotions, Holidays, and Weekends")

# ---------- Historical Data Visualization ----------
st.markdown("---")
st.subheader("📅 Historical vs Predicted Sales")
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.image("Images/actual_vs_predicted_sales_test_set.png", caption="Test Set Performance")
with col_img2:
    st.image("Images/historical_vs_predicted_sales.png", caption="Forecast Extension")

# ---------- Sample Data Table ----------
st.markdown("---")
st.subheader("📋 Sample Historical Data")
sample_df = load_sample_data()
st.dataframe(sample_df.head(10), use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub Repository](https://github.com/avishkarmenge703-netizen/Sales-Forcasting-ml-project)")
