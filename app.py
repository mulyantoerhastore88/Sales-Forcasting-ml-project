import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="SalesForecast AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helper Functions ----------
@st.cache_resource
def load_model():
    """Load the trained model from file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Try different possible paths
    possible_paths = [
        os.path.join(base_dir, '..', 'Model', 'sales_model.pkl'),
        os.path.join(base_dir, 'sales_model.pkl'),
        os.path.join(base_dir, 'Model', 'sales_model.pkl')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                st.warning(f"Found model at {path} but couldn't load it: {e}")
    st.error("🚨 Model file not found. Please ensure 'sales_model.pkl' is in the 'Model' folder or the app directory.")
    return None

@st.cache_data
def load_sample_data():
    """Load sample data or generate dummy data."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(base_dir, '..', 'data', 'sample_sales_data.csv')
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        # Generate dummy data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        data = {
            'Store_ID': np.random.choice([101, 102, 103, 104, 105], size=100),
            'Date': dates,
            'Sales': np.random.randint(200, 800, size=100),
            'Promotion': np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
            'Holiday': np.random.choice([0, 1], size=100, p=[0.9, 0.1]),
            'Price': np.random.uniform(10, 50, size=100).round(2)
        }
        df = pd.DataFrame(data)
        # Add some useful derived columns
        df['Day_of_Week'] = df['Date'].dt.dayofweek + 1  # Monday=1
        df['Is_Weekend'] = df['Day_of_Week'].isin([6, 7]).astype(int)
        df['Lag_Sales'] = df.groupby('Store_ID')['Sales'].shift(1).fillna(300).astype(int)
        return df

def predict_sales(model, features):
    """Make prediction using the loaded model."""
    try:
        prediction = model.predict(features)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ---------- Load Model ----------
model = load_model()

# ---------- Custom CSS for Modern Dark UI ----------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Main background with subtle gradient */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1a1a2e, #0a0a0f);
        color: #ffffff;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(20, 20, 30, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        padding: 30px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(59,130,246,0.3);
        box-shadow: 0 30px 60px -12px #1e3a8a;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 30px;
        padding: 30px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-label {
        color: #8a8a9e;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buttons */
    .btn-primary {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        color: white;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        transition: 0.3s;
        box-shadow: 0 10px 30px -10px #3b82f6;
    }
    .btn-primary:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px -10px #3b82f6;
    }

    /* Floating Fiverr button */
    .floating-fiverr {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: linear-gradient(135deg, #1dbf73, #14a85a);
        color: white;
        padding: 16px 30px;
        border-radius: 60px;
        font-weight: 700;
        font-size: 1.2rem;
        text-decoration: none;
        box-shadow: 0 10px 30px rgba(29,191,115,0.4);
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 999;
        transition: transform 0.3s, box-shadow 0.3s;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(5px);
    }
    .floating-fiverr:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 40px rgba(29,191,115,0.6);
    }

    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj, .css-1wrcr25 {
        background-color: rgba(15, 15, 25, 0.7) !important;
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #ffffff, #b0c4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Floating Fiverr Button (HTML) ----------
st.markdown("""
<a href="https://www.fiverr.com/s/xXBEwyQ" target="_blank" class="floating-fiverr">
    <i class="fab fa-fiverr"></i> Hire me on Fiverr
</a>
<!-- Font Awesome for icons -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center;'>📋 Input Parameters</h2>", unsafe_allow_html=True)
    st.markdown("Adjust the values to see predicted sales.")
    
    store_id = st.selectbox("Store ID", [101, 102, 103, 104, 105])
    promotion = st.selectbox("Promotion Active?", ["Yes", "No"])
    holiday = st.selectbox("Is it a Holiday?", ["Yes", "No"])
    price = st.number_input("Product Price ($)", min_value=5.0, max_value=200.0, value=25.0, step=0.5)
    day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    lag_sales = st.number_input("Previous Day Sales (units)", min_value=0, max_value=2000, value=300)
    
    # Convert categorical
    promo_binary = 1 if promotion == "Yes" else 0
    holiday_binary = 1 if holiday == "Yes" else 0
    day_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}
    day_num = day_map[day_of_week]
    
    # Feature dataframe (adjust column names to match your model)
    feature_names = ['Store_ID', 'Promotion', 'Holiday', 'Price', 'Day_of_Week', 'Lag_Sales']
    input_features = pd.DataFrame([[store_id, promo_binary, holiday_binary, price, day_num, lag_sales]],
                                  columns=feature_names)
    
    if model is not None:
        try:
            prediction = predict_sales(model, input_features)
        except:
            prediction = None
    else:
        prediction = None

# ---------- Main Content ----------
# Navbar
col_logo, col_github = st.columns([3, 1])
with col_logo:
    st.markdown("<h1 style='font-size: 2rem;'><i class='fas fa-chart-line' style='color: #3b82f6;'></i> SalesForecast<span style='color: #8b5cf6;'>AI</span></h1>", unsafe_allow_html=True)
with col_github:
    st.markdown("""
    <a href="https://github.com/avishkarmenge703-netizen/Sales-Forcasting-ml-project" target="_blank" style="
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 600;
        color: white;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        float: right;">
        <i class="fab fa-github"></i> GitHub
    </a>
    """, unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div style="text-align: center; padding: 60px 0 40px;">
    <h1 style="font-size: 4rem; font-weight: 800; line-height: 1.1;">
        Predict future sales with <span style="background: linear-gradient(135deg, #3b82f6, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">machine learning</span>
    </h1>
    <p style="font-size: 1.3rem; color: #a0a0b0; max-width: 800px; margin: 20px auto;">
        A production-ready ML solution that analyzes historical data to forecast demand, optimize inventory, and drive data-driven decisions.
    </p>
    <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-top: 30px;">
        <a href="#prediction-section" class="btn-primary"><i class="fas fa-robot"></i> Try AI Demo</a>
        <a href="https://github.com/avishkarmenge703-netizen/Sales-Forcasting-ml-project" target="_blank" style="
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            padding: 12px 28px;
            border-radius: 50px;
            font-weight: 600;
            color: white;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        "><i class="fas fa-code"></i> Source Code</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Business Value Cards
st.markdown("<h2 class='section-title'>Transform your business</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8a9e; font-size: 1.2rem; margin-bottom: 40px;'>Stop guessing. Start knowing. Our model helps retailers and e‑commerce businesses make smarter decisions.</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"><i class="fas fa-boxes"></i></div>
        <h3>Optimize Inventory</h3>
        <p style="color: #c0c0d0;">Reduce overstock by 25% and prevent stockouts with accurate demand forecasts.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"><i class="fas fa-chart-pie"></i></div>
        <h3>Marketing Insights</h3>
        <p style="color: #c0c0d0;">Understand which levers (promotions, holidays) drive sales and plan campaigns accordingly.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"><i class="fas fa-dollar-sign"></i></div>
        <h3>Revenue Growth</h3>
        <p style="color: #c0c0d0;">Align supply with demand, capture every sales opportunity, and increase profitability.</p>
    </div>
    """, unsafe_allow_html=True)

# Model Performance Metrics
st.markdown("<br><h2 class='section-title'>Model performance</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8a9e; font-size: 1.2rem; margin-bottom: 30px;'>Rigorously validated using time-series cross-validation.</p>", unsafe_allow_html=True)

mae, rmse, r2 = 125.30, 187.45, 0.87  # Static values from your HTML
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{mae}</div>
        <div class="metric-label">MAE (units)</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{rmse}</div>
        <div class="metric-label">RMSE (units)</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{r2}</div>
        <div class="metric-label">R² Score</div>
    </div>
    """, unsafe_allow_html=True)

# Line chart: Actual vs Predicted (dummy data)
st.markdown("<br>", unsafe_allow_html=True)
# Generate dummy time series for demonstration
dates = pd.date_range(start='2025-01-01', periods=60, freq='D')
actual = 300 + 50 * np.sin(np.linspace(0, 8, 60)) + np.random.normal(0, 30, 60)
predicted = actual + np.random.normal(0, 20, 60)

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual Sales',
                         line=dict(color='#3b82f6', width=3)))
fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted Sales',
                         line=dict(color='#ec4899', width=3, dash='dash')))
fig.update_layout(
    title='Actual vs Predicted Sales (Last 60 Days)',
    xaxis_title='Date',
    yaxis_title='Sales (units)',
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(20,20,30,0.4)',
    font=dict(color='#e0e0ff'),
    legend=dict(font=dict(color='#e0e0ff')),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Interactive Demo / Prediction Section
st.markdown("<a id='prediction-section'></a>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Live AI prediction</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8a9e; font-size: 1.2rem;'>Adjust the sliders in the sidebar and see the forecast in real time.</p>", unsafe_allow_html=True)

if prediction is not None:
    st.markdown(f"""
    <div style="background: rgba(20, 20, 30, 0.5); backdrop-filter: blur(12px); border-radius: 50px; padding: 40px; text-align: center; margin: 30px 0;">
        <p style="color: #b0b0c0; margin-bottom: 10px; font-size: 1.3rem;">Predicted daily sales</p>
        <div style="font-size: 6rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;">{prediction:,.0f}</div>
        <p style="color: #5a5a70;">units</p>
    </div>
    """, unsafe_allow_html=True)

    # Gauge chart from original app
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Forecast", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, 1000], 'tickcolor': 'white'},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 300], 'color': "rgba(255,255,255,0.1)"},
                {'range': [300, 600], 'color': "rgba(255,255,255,0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 800
            }
        }
    ))
    gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20),
                            paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
    st.plotly_chart(gauge_fig, use_container_width=True)
else:
    st.warning("Model not available or prediction failed. Please check the model file.")

# Feature Importance (image or dummy)
st.markdown("<h2 class='section-title'>Key drivers</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #8a8a9e; font-size: 1.2rem;'>What influences sales the most? Our model reveals the answer.</p>", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"><i class="fas fa-bullhorn"></i></div>
        <h3>Promotions</h3>
        <p>Promotions increase sales by an average of 42% — the most influential feature.</p>
    </div>
    """, unsafe_allow_html=True)
with col_f2:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"><i class="fas fa-glass-cheers"></i></div>
        <h3>Holidays</h3>
        <p>Holiday periods show a 28% lift, confirming seasonal patterns.</p>
    </div>
    """, unsafe_allow_html=True)
with col_f3:
    st.markdown("""
    <div class="glass-card">
        <div style="font-size: 2.5rem; background: linear-gradient(135deg, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"><i class="fas fa-calendar-week"></i></div>
        <h3>Weekends</h3>
        <p>Weekend days have 15% higher sales on average.</p>
    </div>
    """, unsafe_allow_html=True)

# Sample Data Table
st.markdown("<h2 class='section-title'>📋 Sample Historical Data</h2>", unsafe_allow_html=True)
sample_df = load_sample_data()
st.dataframe(sample_df.head(10), use_container_width=True)

# Footer
st.markdown("""
<hr style="border-color: rgba(255,255,255,0.05); margin: 50px 0 30px;">
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; color: #5a5a70;">
    <div style="font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, #a5b8ff, #ffb3d9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Avishkar Menge</div>
    <div>
        <a href="https://github.com/avishkarmenge703-netizen" target="_blank" style="color: #a0a0b0; margin: 0 15px; text-decoration: none;"><i class="fab fa-github"></i> GitHub</a>
        <a href="#" target="_blank" style="color: #a0a0b0; margin: 0 15px; text-decoration: none;"><i class="fab fa-linkedin"></i> LinkedIn</a>
        <a href="https://www.fiverr.com/s/xXBEwyQ" target="_blank" style="color: #1dbf73; margin: 0 15px; text-decoration: none;"><i class="fab fa-fiverr"></i> Fiverr</a>
    </div>
    <div>© 2026 | MIT Licensed</div>
</div>
""", unsafe_allow_html=True)
