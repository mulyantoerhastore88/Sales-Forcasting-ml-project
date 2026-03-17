# app.py
# Aplikasi Forecasting untuk S&OP - Demand & Supply Planner Senior
# Versi Enterprise (Optimized Performance & Real-world ML Validation)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

# ML & Viz Libraries (Di-load langsung, tidak perlu lazy loading)
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials

# =============================================================================
# KONFIGURASI HALAMAN (HARUS PALING ATAS)
# =============================================================================
st.set_page_config(
    page_title="S&OP Forecast Studio Enterprise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 1rem; }
    .insight-box { background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid #667eea; margin: 1rem 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { height: 3rem; font-size: 1.1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNGSI BANTU (HELPER FUNCTIONS)
# =============================================================================

def safe_date_format(date_series, format_str='%b %y'):
    try:
        valid_dates = date_series.dropna()
        if valid_dates.empty: return "N/A"
        return f"{valid_dates.min().strftime(format_str)} - {valid_dates.max().strftime(format_str)}"
    except Exception as e:
        return "N/A"

@st.cache_data(ttl=3600)
def load_data_from_gsheet():
    """Membaca data dari Google Sheet"""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        if 'gcp_service_account' not in st.secrets:
            st.error("❌ Service account credentials tidak ditemukan di Streamlit secrets")
            return None
            
        credentials = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scope
        )
        gc = gspread.authorize(credentials)
        url = "https://docs.google.com/spreadsheets/d/1PuoII49N-IWOaNO8fSMYGwuvFf1T68_Kez30WN9q8Ds/edit?gid=857579960#gid=857579960"
        
        sh = gc.open_by_url(url)
        data_sources = {}
        
        worksheet = sh.worksheet("Sales")
        data = worksheet.get_all_values()
        
        if data and len(data) > 1:
            headers = data[0]
            df_sales = pd.DataFrame(data[1:], columns=headers)
            data_sources['sales'] = df_sales
            st.success(f"✅ Berhasil membaca {len(df_sales)} baris data sales")
            return data_sources
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Caching diaktifkan untuk mencegah recalculation berat
def clean_and_transform_data(df_sales):
    """Membersihkan dan mentransformasi data"""
    if df_sales is None or df_sales.empty: return None
    
    month_keywords = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_columns = [col for col in df_sales.columns if any(m in str(col) for m in month_keywords)]
    id_columns = [col for col in ['SKU_ID', 'Product Name', 'Sub Brand', 'SKU Tier'] if col in df_sales.columns]
    
    if not month_columns or not id_columns: return None
    
    df_long = pd.melt(df_sales, id_vars=id_columns, value_vars=month_columns, var_name='Period', value_name='Sales')
    df_long['Sales'] = pd.to_numeric(df_long['Sales'], errors='coerce')
    df_long = df_long.dropna(subset=['Sales'])
    df_long = df_long[df_long['Sales'] > 0]
    
    def parse_period_safe(period):
        try:
            for fmt in ['%b %y', '%B %y', '%b-%y', '%m/%Y', '%Y-%m']:
                try: return pd.to_datetime(str(period).strip(), format=fmt)
                except: continue
            return pd.to_datetime(period, errors='coerce')
        except: return pd.NaT

    df_long['Date'] = df_long['Period'].apply(parse_period_safe)
    df_long = df_long.dropna(subset=['Date'])
    
    df_long['Month'] = df_long['Date'].dt.month
    df_long['Year'] = df_long['Date'].dt.year
    df_long = df_long.sort_values(['SKU_ID', 'Date'])
    
    return df_long

def calculate_kpis(df, sku_id=None):
    df_sku = df[df['SKU_ID'] == sku_id].copy() if sku_id else df.copy()
    if len(df_sku) == 0: return None
    
    latest_date = df_sku['Date'].max()
    df_l12m = df_sku[df_sku['Date'] >= (latest_date - pd.DateOffset(months=12))]
    
    mean_val, std_val = df_sku['Sales'].mean(), df_sku['Sales'].std()
    
    yoy_growth = None
    if len(df_sku) >= 24:
        last_12m = df_sku.tail(12)['Sales'].sum()
        prev_12m = df_sku.iloc[-24:-12]['Sales'].sum()
        yoy_growth = ((last_12m - prev_12m) / prev_12m) * 100 if prev_12m != 0 else 0
        
    return {
        'total_sales_l12m': df_l12m['Sales'].sum(),
        'avg_monthly_sales': mean_val,
        'cv': (std_val / mean_val) * 100 if mean_val != 0 else 0,
        'yoy_growth': yoy_growth,
        'seasonality_idx': (df_sku.groupby('Month')['Sales'].mean() / mean_val * 100).to_dict() if mean_val != 0 else {}
    }

def run_forecast_prophet(df_sku, periods=6, changepoint_prior=0.05):
    """Prophet dengan Out-of-Sample Validation"""
    df_prophet = df_sku[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Hitung MAPE Out-of-Sample (Test size: 3 bulan terakhir)
    mape = 100.0
    if len(df_prophet) >= 6:
        train, test = df_prophet.iloc[:-3], df_prophet.iloc[-3:]
        model_eval = Prophet(yearly_seasonality=True, changepoint_prior_scale=changepoint_prior)
        model_eval.fit(train)
        pred_eval = model_eval.predict(model_eval.make_future_dataframe(periods=3, freq='M'))
        actuals = test['y'].values
        predictions = pred_eval.iloc[-3:]['yhat'].values
        if not np.any(actuals == 0):
            mape = mean_absolute_percentage_error(actuals, predictions) * 100

    # Fit Full Data untuk Output Aktual
    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=changepoint_prior)
    model.fit(df_prophet)
    forecast = model.predict(model.make_future_dataframe(periods=periods, freq='M'))
    
    return model, forecast, mape

def run_forecast_xgboost(df_sku, periods=6):
    """XGBoost dengan Out-of-Sample Validation"""
    df_features = df_sku.copy()
    df_features['month'] = df_features['Date'].dt.month
    df_features['lag_1'] = df_features['Sales'].shift(1)
    df_features['lag_2'] = df_features['Sales'].shift(2)
    df_features['rolling_mean_3'] = df_features['Sales'].rolling(window=3, min_periods=1).mean()
    df_features = df_features.dropna()
    
    if len(df_features) < 6: return None, None, 100.0
    
    feature_cols = ['month', 'lag_1', 'lag_2', 'rolling_mean_3']
    
    # Hitung MAPE Out-of-Sample
    train, test = df_features.iloc[:-3], df_features.iloc[-3:]
    model_eval = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model_eval.fit(train[feature_cols], train['Sales'])
    pred_eval = model_eval.predict(test[feature_cols])
    mape = mean_absolute_percentage_error(test['Sales'], pred_eval) * 100 if not np.any(test['Sales'] == 0) else 100.0

    # Fit Full Data
    X, y = df_features[feature_cols], df_features['Sales']
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    # Generate Future
    future_dates = [df_sku['Date'].max() + timedelta(days=30*i) for i in range(1, periods+1)]
    future_preds = []
    last_vals = df_sku['Sales'].tail(3).values
    
    for i in range(periods):
        future_month = future_dates[i].month
        lag_1 = last_vals[-1] if len(last_vals) > 0 else 0
        lag_2 = last_vals[-2] if len(last_vals) > 1 else 0
        rm_3 = np.mean(last_vals[-3:]) if len(last_vals) >= 3 else np.mean(last_vals)
        
        pred = model.predict(np.array([[future_month, lag_1, lag_2, rm_3]]))[0]
        future_preds.append(max(0, pred))
        last_vals = np.append(last_vals, pred)[-3:]

    forecast_df = pd.DataFrame({
        'ds': future_dates, 'yhat': future_preds,
        'yhat_lower': [p * 0.85 for p in future_preds], 'yhat_upper': [p * 1.15 for p in future_preds]
    })
    
    return model, forecast_df, mape

# =============================================================================
# INISIALISASI SESSION STATE
# =============================================================================
for key in ['data_sources', 'data_clean', 'selected_sku', 'forecast_results']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'forecast_results' else {}

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>S&OP Studio</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("🔄 Refresh & Load Data", use_container_width=True, type="primary"):
        with st.spinner("Membaca data..."):
            st.session_state.data_sources = load_data_from_gsheet()
            if st.session_state.data_sources:
                st.session_state.data_clean = clean_and_transform_data(st.session_state.data_sources['sales'])
        st.rerun()

    if st.session_state.data_clean is not None:
        st.subheader("⚙️ Parameter S&OP")
        sku_options = sorted(st.session_state.data_clean['SKU_ID'].unique().tolist())
        selected_sku = st.selectbox("Pilih SKU", ['Semua SKU'] + sku_options)
        st.session_state.selected_sku = selected_sku if selected_sku != 'Semua SKU' else None
        
        forecast_model = st.multiselect("Model ML", ['Prophet', 'XGBoost', 'Ensemble'], default=['Prophet', 'Ensemble'])
        forecast_periods = st.slider("Horizon (bulan)", 3, 12, 6)
        run_forecast = st.button("🚀 Run Multi-Model Forecast", use_container_width=True, type="primary")

# =============================================================================
# MAIN CONTENT & TABS
# =============================================================================
st.markdown("<h1 class='main-header'>📊 S&OP Forecast Studio Enterprise</h1>", unsafe_allow_html=True)

if st.session_state.data_clean is None:
    st.info("👈 Silakan klik 'Refresh & Load Data' di sidebar untuk memulai.")
    st.stop()

tabs = st.tabs(["📈 Data Overview", "🔍 Analytics", "🤖 Multi-Model Forecast", "📋 S&OP Summary", "📤 Export"])

# --- TAB 1: DATA OVERVIEW ---
with tabs[0]:
    st.dataframe(st.session_state.data_clean.head(1000), use_container_width=True)

# --- TAB 2: ANALYTICS ---
with tabs[1]:
    df_eda = st.session_state.data_clean.groupby('Date')['Sales'].sum().reset_index()
    fig = px.line(df_eda, x='Date', y='Sales', title='Total Agregasi Penjualan Seluruh SKU', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: MULTI-MODEL FORECAST ---
with tabs[2]:
    if not st.session_state.selected_sku:
        st.warning("⚠️ Pilih satu SKU spesifik di sidebar untuk melihat hasil forecast.")
    elif 'run_forecast' in locals() and run_forecast:
        with st.spinner("Menjalankan AI Forecasting..."):
            df_sku = st.session_state.data_clean[st.session_state.data_clean['SKU_ID'] == st.session_state.selected_sku]
            results, metrics = {}, {}
            
            if 'Prophet' in forecast_model:
                _, fcst_p, mape_p = run_forecast_prophet(df_sku, periods=forecast_periods)
                results['prophet'] = fcst_p; metrics['Prophet'] = mape_p
            if 'XGBoost' in forecast_model:
                _, fcst_x, mape_x = run_forecast_xgboost(df_sku, periods=forecast_periods)
                if fcst_x is not None: results['xgboost'] = fcst_x; metrics['XGBoost'] = mape_x
            if 'Ensemble' in forecast_model and 'prophet' in results and 'xgboost' in results:
                ens = results['prophet'][['ds']].copy()
                ens['yhat'] = (results['prophet']['yhat'] + results['xgboost']['yhat']) / 2
                results['ensemble'] = ens; metrics['Ensemble'] = (metrics['Prophet'] + metrics['XGBoost']) / 2

            st.success(f"Best Model: **{min(metrics, key=metrics.get)}** (MAPE: {min(metrics.values()):.2f}%)")
            
            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sku['Date'], y=df_sku['Sales'], mode='lines+markers', name='Actual', line=dict(color='black')))
            colors = {'prophet': 'blue', 'xgboost': 'green', 'ensemble': 'red'}
            for m_name, df_f in results.items():
                fig.add_trace(go.Scatter(x=df_f['ds'], y=df_f['yhat'], mode='lines', name=m_name.upper(), line=dict(color=colors[m_name], dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: S&OP SUMMARY (VECTORIZED / NO LOOPING) ---
with tabs[3]:
    st.markdown("<div class='insight-box'><h4>🎯 Executive S&OP Summary (Seluruh SKU)</h4></div>", unsafe_allow_html=True)
    
    with st.spinner("Menghitung rekomendasi stok level Enterprise..."):
        df_all = st.session_state.data_clean.copy()
        
        # 1. Base Forecast (Rata-rata 3 bln terakhir)
        max_dt = df_all['Date'].max()
        df_recent = df_all[df_all['Date'] > (max_dt - pd.DateOffset(months=3))]
        forecast_df = df_recent.groupby('SKU_ID')['Sales'].mean().reset_index().rename(columns={'Sales': 'Forecast'})
        
        # 2. Volatilitas (CV)
        stats_df = df_all.groupby('SKU_ID')['Sales'].agg(['mean', 'std']).reset_index()
        stats_df['CV'] = (stats_df['std'] / stats_df['mean'] * 100).fillna(0)
        
        # 3. Safety Stock (MAD Vectorized)
        df_all = df_all.sort_values(['SKU_ID', 'Date'])
        df_all['MA_3'] = df_all.groupby('SKU_ID')['Sales'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
        df_all['Error'] = abs(df_all['Sales'] - df_all['MA_3'])
        mad_df = df_all.groupby('SKU_ID')['Error'].mean().reset_index().rename(columns={'Error': 'MAD'})
        
        # 4. Merge Data
        summary_df = stats_df.merge(forecast_df, on='SKU_ID', how='left').merge(mad_df, on='SKU_ID', how='left')
        summary_df['Forecast'] = summary_df['Forecast'].fillna(0)
        
        # Safety Factor 1.96 (95% Service Level)
        summary_df['Safety Stock'] = np.where(summary_df['MAD'].notna(), summary_df['MAD'] * 1.96, summary_df['Forecast'] * 0.2)
        summary_df['Total Required'] = summary_df['Forecast'] + summary_df['Safety Stock']
        
        # Styling Status
        def get_status(cv):
            if cv > 50: return "🔴 High Risk"
            elif cv > 25: return "🟡 Medium"
            return "🟢 Stable"
        
        summary_df['Status'] = summary_df['CV'].apply(get_status)
        
        # Mapping Product Name
        if 'Product Name' in df_all.columns:
            prod_map = df_all.drop_duplicates('SKU_ID').set_index('SKU_ID')['Product Name'].to_dict()
            summary_df.insert(1, 'Product', summary_df['SKU_ID'].map(prod_map))
        
        st.dataframe(
            summary_df[['SKU_ID', 'Product', 'Forecast', 'Safety Stock', 'Total Required', 'Status']].round(0),
            use_container_width=True, height=500
        )
        st.session_state.df_export = summary_df

# --- TAB 5: EXPORT ---
with tabs[4]:
    if 'df_export' in st.session_state and st.session_state.df_export is not None:
        csv = st.session_state.df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master S&OP Report",
            data=csv,
            file_name=f"SOP_Report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("Buka tab 'S&OP Summary' terlebih dahulu untuk men-generate data report.")
