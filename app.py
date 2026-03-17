# app.py
# Aplikasi Forecasting untuk S&OP - Demand & Supply Planner Senior
# Versi Enterprise dengan Multiple Models (Prophet, XGBoost, Ensemble)
# Dilengkapi dengan Export ke PPT/Excel dan Advanced Analytics

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

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
# CUSTOM CSS untuk tampilan premium
# =============================================================================
st.markdown("""
<style>
    /* Premium styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNGSI BANTU UNTUK LAZY LOADING
# =============================================================================
@st.cache_resource
def get_prophet():
    """Lazy load Prophet"""
    from prophet import Prophet
    return Prophet

@st.cache_resource
def get_xgboost():
    """Lazy load XGBoost"""
    from xgboost import XGBRegressor
    return XGBRegressor

@st.cache_resource
def get_sklearn():
    """Lazy load sklearn metrics"""
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error

@st.cache_resource
def get_plotly():
    """Lazy load plotly"""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return px, go, make_subplots

@st.cache_resource
def get_matplotlib():
    """Lazy load matplotlib dan seaborn"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns

@st.cache_resource
def get_gspread():
    """Lazy load Google Sheets libraries"""
    import gspread
    from google.oauth2.service_account import Credentials
    return gspread, Credentials

# =============================================================================
# FUNGSI BANTU (HELPER FUNCTIONS)
# =============================================================================

def safe_date_format(date_series, format_str='%b %y'):
    """
    Format tanggal dengan aman, handle NaT values
    """
    try:
        if date_series is None or date_series.empty:
            return "N/A"
        
        valid_dates = date_series.dropna()
        if valid_dates.empty:
            return "N/A"
        
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        
        if pd.isna(min_date) or pd.isna(max_date):
            return "N/A"
        
        return f"{min_date.strftime(format_str)} - {max_date.strftime(format_str)}"
    except Exception as e:
        return f"Error: {str(e)[:50]}"

@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def load_data_from_gsheet():
    """
    Membaca data dari Google Sheet menggunakan service account
    """
    try:
        gspread, Credentials = get_gspread()
        
        # Ambil credentials dari Streamlit secrets
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        # Cek apakah secrets tersedia
        if 'gcp_service_account' not in st.secrets:
            st.error("❌ Service account credentials tidak ditemukan di Streamlit secrets")
            st.info("""
            Cara setting secrets:
            1. Buka https://share.streamlit.io
            2. Pilih app ini
            3. Settings → Secrets
            4. Copy paste JSON service account Anda
            """)
            return None
        
        try:
            credentials = Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"]),
                scopes=scope
            )
        except Exception as e:
            st.error(f"❌ Error membuat credentials: {str(e)}")
            return None
        
        gc = gspread.authorize(credentials)
        
        # Buka spreadsheet
        url = "https://docs.google.com/spreadsheets/d/1PuoII49N-IWOaNO8fSMYGwuvFf1T68_Kez30WN9q8Ds/edit?gid=857579960#gid=857579960"
        
        try:
            sh = gc.open_by_url(url)
        except Exception as e:
            st.error(f"❌ Tidak bisa membuka Google Sheet: {str(e)}")
            st.info("Pastikan service account email sudah diberi akses ke spreadsheet")
            return None
        
        # Ambil semua worksheet yang relevan
        data_sources = {}
        
        # Baca Sales Data
        try:
            worksheet = sh.worksheet("Sales")
            data = worksheet.get_all_values()
            
            if not data or len(data) < 2:
                st.error("Sheet 'Sales' kosong atau tidak memiliki data")
                return None
            
            headers = data[0]
            rows = data[1:]
            
            # Validasi headers
            if 'SKU_ID' not in headers:
                st.error("Kolom 'SKU_ID' tidak ditemukan di sheet Sales")
                return None
            
            df_sales = pd.DataFrame(rows, columns=headers)
            st.success(f"✅ Berhasil membaca {len(rows)} baris data sales")
            data_sources['sales'] = df_sales
            
        except Exception as e:
            st.error(f"Error membaca sheet Sales: {str(e)}")
            return None
        
        # Baca Stock On Hand (opsional)
        try:
            worksheet = sh.worksheet("Stock_Onhand")
            data = worksheet.get_all_values()
            if data and len(data) > 1:
                headers = data[0]
                rows = data[1:]
                data_sources['stock'] = pd.DataFrame(rows, columns=headers)
        except:
            data_sources['stock'] = None
        
        # Baca Product Master (opsional)
        try:
            worksheet = sh.worksheet("Product_Master")
            data = worksheet.get_all_values()
            if data and len(data) > 1:
                headers = data[0]
                rows = data[1:]
                data_sources['master'] = pd.DataFrame(rows, columns=headers)
        except:
            data_sources['master'] = None
        
        return data_sources
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_and_transform_data(df_sales):
    """
    Membersihkan dan mentransformasi data dari format lebar ke format panjang
    """
    if df_sales is None or df_sales.empty:
        st.error("Data sales kosong")
        return None
    
    # Identifikasi kolom bulan (format: "Jan 24", "Feb 24", dll)
    month_keywords = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    month_columns = []
    for col in df_sales.columns:
        if any(month in str(col) for month in month_keywords):
            month_columns.append(col)
    
    if not month_columns:
        st.error("Tidak ditemukan kolom bulan dalam format yang benar")
        st.info("Kolom yang ditemukan: " + ", ".join(df_sales.columns[:10].tolist()))
        return None
    
    # Pastikan kolom identifier ada
    id_columns = ['SKU_ID', 'Product Name', 'Sub Brand', 'SKU Tier']
    available_ids = []
    for col in id_columns:
        if col in df_sales.columns:
            available_ids.append(col)
    
    if not available_ids:
        st.error("Tidak ditemukan kolom identifier (SKU_ID, dll)")
        return None
    
    st.info(f"📊 Ditemukan {len(month_columns)} kolom bulan")
    st.info(f"📋 Identifier columns: {available_ids}")
    
    # Melt data dari format lebar ke panjang
    df_long = pd.melt(
        df_sales,
        id_vars=available_ids,
        value_vars=month_columns,
        var_name='Period',
        value_name='Sales'
    )
    
    # Konversi Sales ke numeric, error jadi NaN
    df_long['Sales'] = pd.to_numeric(df_long['Sales'], errors='coerce')
    
    # Drop baris dengan Sales NaN atau 0
    df_long = df_long.dropna(subset=['Sales'])
    df_long = df_long[df_long['Sales'] > 0]
    
    if df_long.empty:
        st.error("Tidak ada data sales yang valid setelah cleaning")
        return None
    
    # Fungsi untuk konversi period ke date dengan penanganan error yang lebih baik
    def parse_period_safe(period):
        try:
            if not period or pd.isna(period):
                return pd.NaT
            
            # Bersihkan string
            period = str(period).strip()
            
            # Coba beberapa format
            formats = ['%b %y', '%B %y', '%b-%y', '%B-%y', '%m/%Y', '%Y-%m']
            
            for fmt in formats:
                try:
                    return pd.to_datetime(period, format=fmt)
                except:
                    continue
            
            # Fallback ke parser umum
            return pd.to_datetime(period, errors='coerce')
        except:
            return pd.NaT
    
    df_long['Date'] = df_long['Period'].apply(parse_period_safe)
    
    # Drop baris dengan Date NaT
    initial_count = len(df_long)
    df_long = df_long.dropna(subset=['Date'])
    final_count = len(df_long)
    
    if final_count == 0:
        st.error("Semua data gagal diparse menjadi tanggal")
        st.info(f"Contoh periode: {df_long['Period'].iloc[0] if not df_long.empty else 'N/A'}")
        return None
    
    if final_count < initial_count:
        st.warning(f"{initial_count - final_count} baris di-drop karena format tanggal tidak valid")
    
    # Tambah kolom bulan dan tahun untuk analisis
    df_long['Month'] = df_long['Date'].dt.month
    df_long['Year'] = df_long['Date'].dt.year
    df_long['MonthName'] = df_long['Date'].dt.strftime('%b')
    df_long['Quarter'] = df_long['Date'].dt.quarter
    
    # Sort by SKU and Date
    df_long = df_long.sort_values(['SKU_ID', 'Date'])
    
    st.success(f"✅ Berhasil transformasi: {len(df_long)} records dari {df_long['Date'].min().year} - {df_long['Date'].max().year}")
    
    return df_long

def calculate_kpis(df, sku_id=None):
    """
    Menghitung KPI untuk SKU tertentu atau keseluruhan
    """
    if sku_id:
        df_sku = df[df['SKU_ID'] == sku_id].copy()
    else:
        df_sku = df.copy()
    
    if len(df_sku) == 0:
        return None
    
    # Total Sales Last 12 Months
    latest_date = df_sku['Date'].max()
    one_year_ago = latest_date - pd.DateOffset(months=12)
    df_l12m = df_sku[df_sku['Date'] >= one_year_ago]
    total_sales_l12m = df_l12m['Sales'].sum()
    
    # Average Monthly Sales
    avg_monthly_sales = df_sku['Sales'].mean()
    
    # Coefficient of Variation (CV)
    mean = df_sku['Sales'].mean()
    std = df_sku['Sales'].std()
    cv = (std / mean) * 100 if mean != 0 else 0
    
    # Growth (YoY)
    if len(df_sku) >= 24:
        last_12m = df_sku.tail(12)['Sales'].sum()
        prev_12m = df_sku.iloc[-24:-12]['Sales'].sum()
        yoy_growth = ((last_12m - prev_12m) / prev_12m) * 100 if prev_12m != 0 else 0
    else:
        yoy_growth = None
    
    # Seasonality Index
    monthly_avg = df_sku.groupby('Month')['Sales'].mean()
    overall_avg = df_sku['Sales'].mean()
    seasonality_idx = (monthly_avg / overall_avg * 100).to_dict() if overall_avg != 0 else {}
    
    return {
        'total_sales_l12m': total_sales_l12m,
        'avg_monthly_sales': avg_monthly_sales,
        'cv': cv,
        'yoy_growth': yoy_growth,
        'seasonality_idx': seasonality_idx
    }

def run_forecast_prophet(df_sku, periods=6, changepoint_prior=0.05, seasonality_mode='multiplicative'):
    """
    Forecasting dengan Prophet
    """
    Prophet = get_prophet()
    mean_absolute_percentage_error = get_sklearn()
    
    # Siapkan data untuk Prophet
    df_prophet = df_sku[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Inisialisasi dan fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior,
        seasonality_mode=seasonality_mode
    )
    
    # Tambahkan monthly seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(df_prophet)
    
    # Buat dataframe future
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    # Hitung akurasi (in-sample)
    predictions = forecast.iloc[:len(df_prophet)]['yhat']
    actuals = df_prophet['y'].values
    
    # Handle division by zero
    if np.any(actuals == 0):
        mape = 100.0
    else:
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
    
    return model, forecast, mape

def run_forecast_xgboost(df_sku, periods=6):
    """
    Forecasting dengan XGBoost (menggunakan fitur time series)
    """
    XGBRegressor = get_xgboost()
    mean_absolute_percentage_error = get_sklearn()
    
    # Buat fitur dari tanggal
    df_features = df_sku.copy()
    df_features['month'] = df_features['Date'].dt.month
    df_features['quarter'] = df_features['Date'].dt.quarter
    df_features['year'] = df_features['Date'].dt.year
    df_features['dayofyear'] = df_features['Date'].dt.dayofyear
    df_features['lag_1'] = df_features['Sales'].shift(1)
    df_features['lag_2'] = df_features['Sales'].shift(2)
    df_features['lag_3'] = df_features['Sales'].shift(3)
    df_features['rolling_mean_3'] = df_features['Sales'].rolling(window=3, min_periods=1).mean()
    
    # Drop NA
    df_features = df_features.dropna()
    
    if len(df_features) < 5:
        # Not enough data for XGBoost
        return None, None, 100.0
    
    # Siapkan X dan y
    feature_cols = ['month', 'quarter', 'year', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3']
    X = df_features[feature_cols]
    y = df_features['Sales']
    
    # Train model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)
    
    # Prediksi in-sample untuk MAPE
    predictions = model.predict(X)
    
    # Handle division by zero
    if np.any(y == 0):
        mape = 100.0
    else:
        mape = mean_absolute_percentage_error(y, predictions) * 100
    
    # Generate future dates
    last_date = df_sku['Date'].max()
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
    
    # Buat prediksi untuk masa depan
    future_predictions = []
    last_values = df_sku['Sales'].tail(3).values
    
    for i in range(periods):
        # Buat fitur untuk prediksi
        future_month = future_dates[i].month
        future_quarter = (future_month - 1) // 3 + 1
        future_year = future_dates[i].year
        
        # Gunakan nilai terakhir sebagai lag
        lag_1 = last_values[-1] if len(last_values) > 0 else 0
        lag_2 = last_values[-2] if len(last_values) > 1 else 0
        lag_3 = last_values[-3] if len(last_values) > 2 else 0
        rolling_mean = np.mean(last_values[-3:]) if len(last_values) >= 3 else np.mean(last_values) if len(last_values) > 0 else 0
        
        # Buat array fitur
        features = np.array([[future_month, future_quarter, future_year, 
                             lag_1, lag_2, lag_3, rolling_mean]])
        
        # Prediksi
        pred = model.predict(features)[0]
        future_predictions.append(max(0, pred))  # Ensure non-negative
        
        # Update last_values untuk iterasi berikutnya
        last_values = np.append(last_values[1:], pred) if len(last_values) >= 3 else np.append(last_values, pred)
    
    # Buat dataframe forecast
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions,
        'yhat_lower': [p * 0.85 for p in future_predictions],  # Simple CI
        'yhat_upper': [p * 1.15 for p in future_predictions]
    })
    
    return model, forecast_df, mape

def create_ensemble_forecast(forecast_prophet, forecast_xgb, weights=None):
    """
    Membuat ensemble forecast dengan bobot
    """
    if weights is None:
        weights = {'prophet': 0.5, 'xgboost': 0.5}
    
    # Gabungkan forecast
    ensemble = forecast_prophet[['ds']].copy()
    ensemble['yhat'] = (weights['prophet'] * forecast_prophet['yhat'] + 
                        weights['xgboost'] * forecast_xgb['yhat'])
    ensemble['yhat_lower'] = (weights['prophet'] * forecast_prophet['yhat_lower'] + 
                             weights['xgboost'] * forecast_xgb['yhat_lower'])
    ensemble['yhat_upper'] = (weights['prophet'] * forecast_prophet['yhat_upper'] + 
                             weights['xgboost'] * forecast_xgb['yhat_upper'])
    
    return ensemble

def calculate_safety_stock(df_sku, forecast_value, service_level=0.95):
    """
    Menghitung safety stock berdasarkan MAD (Mean Absolute Deviation)
    """
    df_sku = df_sku.sort_values('Date')
    df_sku['MA_3'] = df_sku['Sales'].rolling(window=3, min_periods=1).mean().shift(1)
    df_sku['Error'] = abs(df_sku['Sales'] - df_sku['MA_3'])
    
    mad = df_sku['Error'].mean()
    
    # Faktor keamanan untuk service level
    safety_factors = {0.90: 1.28, 0.95: 1.96, 0.99: 2.33}
    safety_factor = safety_factors.get(service_level, 1.96)
    
    safety_stock = safety_factor * mad
    
    return safety_stock if not np.isnan(safety_stock) else forecast_value * 0.2  # Default 20% jika MAD tidak tersedia

# =============================================================================
# INISIALISASI SESSION STATE
# =============================================================================
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = None
if 'data_clean' not in st.session_state:
    st.session_state.data_clean = None
if 'selected_sku' not in st.session_state:
    st.session_state.selected_sku = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

# =============================================================================
# SIDEBAR - NAVIGASI DAN KONTROL
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bar-chart.png", width=80)
    st.markdown("<h1 style='text-align: center;'>S&OP Studio</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Status koneksi
    if st.session_state.data_clean is not None and not st.session_state.data_clean.empty:
        st.success("✅ Terhubung ke Google Sheets")
        
        # Info data yang aman
        try:
            total_sku = st.session_state.data_clean['SKU_ID'].nunique() if 'SKU_ID' in st.session_state.data_clean.columns else 0
            total_records = len(st.session_state.data_clean)
            st.info(f"📊 {total_sku} SKU | {total_records} records")
            
            # Date range yang aman
            if 'Date' in st.session_state.data_clean.columns:
                date_range = safe_date_format(st.session_state.data_clean['Date'])
                st.caption(f"📅 {date_range}")
        except:
            st.info("📊 Data tersedia")
    else:
        st.warning("⏳ Menunggu koneksi...")
    
    st.markdown("---")
    
    # Tombol load data
    if st.button("🔄 Refresh & Load Data", use_container_width=True, type="primary"):
        with st.spinner("Membaca data dari Google Sheets..."):
            st.session_state.data_sources = load_data_from_gsheet()
            if st.session_state.data_sources and 'sales' in st.session_state.data_sources:
                with st.spinner("Transformasi data..."):
                    st.session_state.data_clean = clean_and_transform_data(
                        st.session_state.data_sources['sales']
                    )
                st.rerun()
    
    st.markdown("---")
    
    # Filter jika data sudah ada
    if st.session_state.data_clean is not None and not st.session_state.data_clean.empty:
        st.subheader("🔍 Filter Data")
        
        # Filter by Sub Brand
        brand_options = ['Semua Brand']
        if 'Sub Brand' in st.session_state.data_clean.columns:
            brands = st.session_state.data_clean['Sub Brand'].dropna().unique().tolist()
            if brands:
                brand_options.extend(sorted(brands))
        selected_brand = st.selectbox("Sub Brand", brand_options)
        
        # Filter by SKU Tier
        tier_options = ['Semua Tier']
        if 'SKU Tier' in st.session_state.data_clean.columns:
            tiers = st.session_state.data_clean['SKU Tier'].dropna().unique().tolist()
            if tiers:
                tier_options.extend(sorted(tiers))
        selected_tier = st.selectbox("SKU Tier", tier_options)
        
        # Filter SKU berdasarkan brand dan tier
        df_filtered = st.session_state.data_clean.copy()
        if selected_brand != 'Semua Brand' and 'Sub Brand' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Sub Brand'] == selected_brand]
        if selected_tier != 'Semua Tier' and 'SKU Tier' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['SKU Tier'] == selected_tier]
        
        sku_options = ['Semua SKU']
        if 'SKU_ID' in df_filtered.columns:
            skus = df_filtered['SKU_ID'].unique().tolist()
            if skus:
                sku_options.extend(sorted(skus))
        selected_sku = st.selectbox("Pilih SKU", sku_options)
        
        st.session_state.selected_sku = selected_sku if selected_sku != 'Semua SKU' else None
        
        st.markdown("---")
        
        # Parameter forecasting
        st.subheader("⚙️ Forecast Parameters")
        
        forecast_model = st.multiselect(
            "Model",
            ['Prophet', 'XGBoost', 'Ensemble'],
            default=['Prophet', 'Ensemble']
        )
        
        forecast_periods = st.slider("Horizon (bulan)", 3, 12, 6)
        confidence_level = st.select_slider(
            "Service Level",
            options=[0.90, 0.95, 0.99],
            value=0.95
        )
        
        if 'Prophet' in forecast_model:
            changepoint_prior = st.slider(
                "Prophet: Trend Sensitivity",
                0.01, 0.5, 0.05, 0.01,
                help="Nilai lebih besar = lebih sensitif terhadap perubahan trend"
            )
        
        st.markdown("---")
        
        # Tombol run forecast
        run_forecast = st.button("🚀 Run Forecast", use_container_width=True, type="primary")

# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("<h1 class='main-header'>📊 S&OP Forecast Studio Enterprise</h1>", unsafe_allow_html=True)

if st.session_state.data_clean is None or st.session_state.data_clean.empty:
    # Tampilan welcome
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.icons8.com/fluency/96/000000/bar-chart.png", width=200)
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>Selamat Datang di S&OP Forecast Studio</h2>
            <p style='color: #666; font-size: 1.1rem;'>
                Aplikasi forecasting untuk Demand & Supply Planner<br>
                dengan dukungan Multiple Models dan Advanced Analytics
            </p>
            <p style='margin-top: 2rem;'>
                👈 Klik <b>'Refresh & Load Data'</b> di sidebar untuk memulai
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# Tabs untuk navigasi
tabs = st.tabs([
    "📈 Data Overview", 
    "🔍 EDA & Analytics", 
    "🤖 Multi-Model Forecast",
    "📊 Model Comparison",
    "📋 S&OP Summary",
    "📤 Export & Report"
])

# =============================================================================
# TAB 1: DATA OVERVIEW
# =============================================================================

with tabs[0]:
    st.header("📈 Data Overview")
    
    # Hitung metrics dengan aman
    total_sku = st.session_state.data_clean['SKU_ID'].nunique() if 'SKU_ID' in st.session_state.data_clean.columns else 0
    total_brand = st.session_state.data_clean['Sub Brand'].nunique() if 'Sub Brand' in st.session_state.data_clean.columns else 0
    total_sales = st.session_state.data_clean['Sales'].sum() if 'Sales' in st.session_state.data_clean.columns else 0
    date_range = safe_date_format(st.session_state.data_clean['Date']) if 'Date' in st.session_state.data_clean.columns else "N/A"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total SKU", total_sku)
    with col2:
        st.metric("Total Sub Brand", total_brand)
    with col3:
        st.metric("Periode Data", date_range)
    with col4:
        st.metric("Total Sales", f"{total_sales:,.0f}")
    
    st.markdown("---")
    
    # Data preview
    view_option = st.radio(
        "Tampilan Data:",
        ["Long Format (Time Series)", "Wide Format (Pivot Table)"],
        horizontal=True
    )
    
    if view_option == "Long Format (Time Series)":
        # Pilih kolom yang ada
        display_cols = []
        for col in ['SKU_ID', 'Product Name', 'Sub Brand', 'Date', 'Sales']:
            if col in st.session_state.data_clean.columns:
                display_cols.append(col)
        
        st.dataframe(
            st.session_state.data_clean[display_cols].head(1000),
            use_container_width=True,
            height=400
        )
        st.caption(f"Menampilkan 1000 dari {len(st.session_state.data_clean)} records")
    else:
        # Pivot untuk tampilan wide
        try:
            index_cols = []
            for col in ['SKU_ID', 'Product Name', 'Sub Brand', 'SKU Tier']:
                if col in st.session_state.data_clean.columns:
                    index_cols.append(col)
            
            if index_cols and 'Date' in st.session_state.data_clean.columns:
                pivot_data = st.session_state.data_clean.pivot_table(
                    index=index_cols,
                    columns='Date',
                    values='Sales',
                    aggfunc='sum'
                ).round(0)
                
                # Format kolom tanggal
                if not pivot_data.empty and hasattr(pivot_data.columns, 'strftime'):
                    pivot_data.columns = [col.strftime('%b %y') for col in pivot_data.columns]
                
                st.dataframe(pivot_data.head(100), use_container_width=True, height=400)
            else:
                st.warning("Kolom untuk pivot table tidak lengkap")
                st.dataframe(st.session_state.data_clean.head(100))
        except Exception as e:
            st.error(f"Error membuat pivot table: {str(e)}")
            st.dataframe(st.session_state.data_clean.head(100))
    
    # Download button
    csv = st.session_state.data_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Raw Data (CSV)",
        data=csv,
        file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# =============================================================================
# TAB 2: EDA & ANALYTICS
# =============================================================================

with tabs[1]:
    st.header("🔍 Exploratory Data Analysis")
    
    # Filter untuk EDA
    col1, col2 = st.columns(2)
    with col1:
        sku_list = ['Semua SKU (Aggregated)']
        if 'SKU_ID' in st.session_state.data_clean.columns:
            sku_list.extend(sorted(st.session_state.data_clean['SKU_ID'].unique().tolist()))
        eda_sku = st.selectbox("Pilih SKU untuk Analisis Detail", sku_list)
    
    with col2:
        chart_type = st.selectbox(
            "Tipe Chart",
            ['Line Chart', 'Bar Chart', 'Area Chart', 'Box Plot']
        )
    
    # Filter data
    if eda_sku != 'Semua SKU (Aggregated)' and 'SKU_ID' in st.session_state.data_clean.columns:
        df_eda = st.session_state.data_clean[
            st.session_state.data_clean['SKU_ID'] == eda_sku
        ].copy()
        title_prefix = f"SKU: {eda_sku}"
    else:
        df_eda = st.session_state.data_clean.groupby('Date')['Sales'].sum().reset_index()
        df_eda['SKU_ID'] = 'TOTAL'
        title_prefix = "Semua SKU (Aggregated)"
    
    # KPI Cards
    kpis = calculate_kpis(df_eda)
    
    if kpis:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sales L12M", f"{kpis['total_sales_l12m']:,.0f}")
        with col2:
            st.metric("Rata-rata Bulanan", f"{kpis['avg_monthly_sales']:,.0f}")
        with col3:
            st.metric("Volatilitas (CV)", f"{kpis['cv']:.1f}%")
        with col4:
            if kpis['yoy_growth'] is not None:
                st.metric("YoY Growth", f"{kpis['yoy_growth']:.1f}%")
            else:
                st.metric("YoY Growth", "N/A")
    
    st.markdown("---")
    
    # Time Series Plot
    st.subheader(f"📅 {title_prefix}")
    
    # Lazy load plotly
    px, go, make_subplots = get_plotly()
    
    if chart_type == 'Line Chart':
        fig = px.line(
            df_eda, 
            x='Date', 
            y='Sales',
            title=f'Tren Penjualan - {title_prefix}',
            markers=True
        )
    elif chart_type == 'Bar Chart':
        fig = px.bar(
            df_eda, 
            x='Date', 
            y='Sales',
            title=f'Tren Penjualan - {title_prefix}'
        )
    elif chart_type == 'Area Chart':
        fig = px.area(
            df_eda, 
            x='Date', 
            y='Sales',
            title=f'Tren Penjualan - {title_prefix}'
        )
    else:  # Box Plot
        df_eda['Year-Month'] = df_eda['Date'].dt.strftime('%Y-%m')
        fig = px.box(
            df_eda, 
            x='Year-Month', 
            y='Sales',
            title=f'Distribusi Penjualan per Bulan - {title_prefix}'
        )
    
    fig.update_layout(
        xaxis_title="Periode",
        yaxis_title="Sales (Unit)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analisis Musiman
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Seasonal Pattern")
        
        # Monthly pattern
        monthly_pattern = df_eda.groupby('Month')['Sales'].agg(['mean', 'std']).reset_index()
        monthly_pattern['MonthName'] = monthly_pattern['Month'].apply(lambda x: calendar.month_abbr[x])
        
        fig_seasonal = px.bar(
            monthly_pattern,
            x='MonthName',
            y='mean',
            error_y='std',
            title='Rata-rata Penjualan per Bulan',
            labels={'mean': 'Average Sales', 'MonthName': 'Bulan'}
        )
        fig_seasonal.update_layout(height=400)
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Seasonality Index")
        
        if kpis and 'seasonality_idx' in kpis and kpis['seasonality_idx']:
            season_df = pd.DataFrame(
                list(kpis['seasonality_idx'].items()),
                columns=['Month', 'Index']
            )
            season_df['MonthName'] = season_df['Month'].apply(lambda x: calendar.month_abbr[int(x)])
            season_df = season_df.sort_values('Month')
            
            fig_idx = px.bar(
                season_df,
                x='MonthName',
                y='Index',
                title='Seasonality Index (100% = Rata-rata)',
                color='Index',
                color_continuous_scale='RdYlGn'
            )
            fig_idx.add_hline(y=100, line_dash="dash", line_color="red")
            fig_idx.update_layout(height=400)
            st.plotly_chart(fig_idx, use_container_width=True)

# =============================================================================
# TAB 3: MULTI-MODEL FORECAST
# =============================================================================

with tabs[2]:
    st.header("🤖 Multi-Model Forecasting")
    
    if not st.session_state.selected_sku:
        st.warning("⚠️ Silakan pilih satu SKU spesifik di sidebar untuk melakukan forecasting")
    else:
        # Siapkan data untuk SKU terpilih
        df_sku = st.session_state.data_clean[
            st.session_state.data_clean['SKU_ID'] == st.session_state.selected_sku
        ].copy()
        
        if df_sku.empty:
            st.error(f"Tidak ada data untuk SKU: {st.session_state.selected_sku}")
            st.stop()
        
        # Tampilkan info SKU
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**SKU:** {st.session_state.selected_sku}")
        with col2:
            product_name = "N/A"
            if 'Product Name' in st.session_state.data_clean.columns:
                product_names = st.session_state.data_clean[
                    st.session_state.data_clean['SKU_ID'] == st.session_state.selected_sku
                ]['Product Name'].unique()
                if len(product_names) > 0:
                    product_name = product_names[0]
            st.info(f"**Product:** {product_name}")
        with col3:
            sub_brand = "N/A"
            if 'Sub Brand' in st.session_state.data_clean.columns:
                sub_brands = st.session_state.data_clean[
                    st.session_state.data_clean['SKU_ID'] == st.session_state.selected_sku
                ]['Sub Brand'].unique()
                if len(sub_brands) > 0:
                    sub_brand = sub_brands[0]
            st.info(f"**Sub Brand:** {sub_brand}")
        
        st.markdown("---")
        
        # Cek apakah user sudah klik run forecast
        if 'run_forecast' in locals() and run_forecast and st.session_state.selected_sku:
            with st.spinner("Menjalankan multiple forecasting models..."):
                
                results = {}
                metrics = {}
                
                # Prophet Forecast
                if 'Prophet' in forecast_model:
                    try:
                        model_p, forecast_p, mape_p = run_forecast_prophet(
                            df_sku, 
                            periods=forecast_periods,
                            changepoint_prior=changepoint_prior if 'changepoint_prior' in locals() else 0.05
                        )
                        results['prophet'] = {
                            'model': model_p,
                            'forecast': forecast_p,
                            'mape': mape_p
                        }
                        metrics['Prophet'] = mape_p
                        st.success("✅ Prophet selesai")
                    except Exception as e:
                        st.error(f"Prophet error: {str(e)}")
                
                # XGBoost Forecast
                if 'XGBoost' in forecast_model:
                    try:
                        model_x, forecast_x, mape_x = run_forecast_xgboost(
                            df_sku,
                            periods=forecast_periods
                        )
                        if model_x is not None:
                            results['xgboost'] = {
                                'model': model_x,
                                'forecast': forecast_x,
                                'mape': mape_x
                            }
                            metrics['XGBoost'] = mape_x
                            st.success("✅ XGBoost selesai")
                    except Exception as e:
                        st.error(f"XGBoost error: {str(e)}")
                
                # Ensemble Forecast
                if 'Ensemble' in forecast_model and 'prophet' in results and 'xgboost' in results:
                    try:
                        ensemble = create_ensemble_forecast(
                            results['prophet']['forecast'],
                            results['xgboost']['forecast']
                        )
                        results['ensemble'] = {
                            'forecast': ensemble,
                            'mape': np.mean([metrics['Prophet'], metrics['XGBoost']])
                        }
                        metrics['Ensemble'] = np.mean([metrics['Prophet'], metrics['XGBoost']])
                        st.success("✅ Ensemble selesai")
                    except Exception as e:
                        st.error(f"Ensemble error: {str(e)}")
                
                # Simpan hasil
                st.session_state.forecast_results[st.session_state.selected_sku] = {
                    'results': results,
                    'metrics': metrics,
                    'params': {
                        'periods': forecast_periods,
                        'confidence_level': confidence_level,
                        'models': forecast_model
                    },
                    'historical': df_sku
                }
        
        # Tampilkan hasil forecast jika ada
        if st.session_state.selected_sku in st.session_state.forecast_results:
            result = st.session_state.forecast_results[st.session_state.selected_sku]
            
            if not result['results']:
                st.warning("Tidak ada model yang berhasil dijalankan")
                st.stop()
            
            # Model Performance Metrics
            st.subheader("📊 Model Performance Comparison")
            
            metrics_df = pd.DataFrame(
                list(result['metrics'].items()),
                columns=['Model', 'MAPE (%)']
            ).round(2)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(metrics_df, use_container_width=True)
                
                # Best model
                if not metrics_df.empty:
                    best_idx = metrics_df['MAPE (%)'].idxmin()
                    best_model = metrics_df.loc[best_idx]
                    st.success(f"🏆 Best Model: **{best_model['Model']}** dengan MAPE {best_model['MAPE (%)']:.1f}%")
            
            with col2:
                # Bar chart perbandingan
                px, go, _ = get_plotly()
                fig_metrics = px.bar(
                    metrics_df,
                    x='Model',
                    y='MAPE (%)',
                    title='Model Accuracy Comparison (Lower is Better)',
                    color='MAPE (%)',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            st.markdown("---")
            
            # Plot semua forecast
            st.subheader("📈 Forecast Comparison")
            
            px, go, _ = get_plotly()
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=result['historical']['Date'],
                y=result['historical']['Sales'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='black', width=2)
            ))
            
            # Plot each model's forecast
            colors = {'prophet': 'blue', 'xgboost': 'green', 'ensemble': 'red'}
            names = {'prophet': 'Prophet', 'xgboost': 'XGBoost', 'ensemble': 'Ensemble'}
            
            for model_name in result['results'].keys():
                forecast = result['results'][model_name]['forecast']
                
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=names.get(model_name, model_name),
                    line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash' if model_name != 'ensemble' else 'solid')
                ))
                
                # Add confidence interval only for ensemble
                if model_name == 'ensemble':
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'CI {int(result["params"]["confidence_level"]*100)}%'
                    ))
            
            fig.update_layout(
                title=f'Forecast Comparison - {st.session_state.selected_sku}',
                xaxis_title='Periode',
                yaxis_title='Sales',
                hovermode='x unified',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 4: MODEL COMPARISON (untuk multiple SKU)
# =============================================================================

with tabs[3]:
    st.header("📊 Model Comparison Across SKUs")
    
    if st.session_state.forecast_results:
        # Buat comparison dataframe
        comparison_data = []
        
        for sku, result in st.session_state.forecast_results.items():
            row = {'SKU': sku}
            if 'metrics' in result:
                row.update(result['metrics'])
            comparison_data.append(row)
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Tampilkan tabel perbandingan
            st.subheader("Detail Comparison")
            st.dataframe(df_comparison.round(2), use_container_width=True)
            
            # Best model per SKU
            st.subheader("🏆 Best Model per SKU")
            best_models = []
            for _, row in df_comparison.iterrows():
                sku = row['SKU']
                metrics = {k: v for k, v in row.items() if k != 'SKU' and pd.notna(v)}
                if metrics:
                    best_model = min(metrics, key=metrics.get)
                    best_models.append({
                        'SKU': sku,
                        'Best Model': best_model,
                        'MAPE': metrics[best_model]
                    })
            
            if best_models:
                st.dataframe(pd.DataFrame(best_models), use_container_width=True)
    else:
        st.info("Belum ada hasil forecast. Jalankan forecast di tab sebelumnya.")

# =============================================================================
# TAB 5: S&OP SUMMARY
# =============================================================================

with tabs[4]:
    st.header("📋 S&OP Meeting Summary")
    
    st.markdown("""
    <div class='insight-box'>
        <h4>🎯 Ringkasan untuk Meeting S&OP Bulanan</h4>
        <p>Berdasarkan hasil forecast dan analisis stok untuk semua SKU</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Buat summary untuk semua SKU (batasi 50 untuk performa)
    all_skus = st.session_state.data_clean['SKU_ID'].unique()[:50]
    
    summary_data = []
    
    with st.spinner("Menghitung rekomendasi untuk SKU..."):
        for sku in all_skus:
            df_sku = st.session_state.data_clean[
                st.session_state.data_clean['SKU_ID'] == sku
            ].copy()
            
            # Simple forecast (average of last 3 months)
            last_3_months = df_sku.sort_values('Date').tail(3)['Sales'].mean()
            
            # Safety stock
            safety_stock = calculate_safety_stock(df_sku, last_3_months, service_level=0.95)
            
            # Hitung CV untuk klasifikasi
            mean = df_sku['Sales'].mean()
            std = df_sku['Sales'].std()
            cv = (std / mean) * 100 if mean != 0 else 0
            
            # Tentukan status
            if cv > 50:
                volatility = "High"
                status_color = "🔴"
            elif cv > 25:
                volatility = "Medium"
                status_color = "🟡"
            else:
                volatility = "Low"
                status_color = "🟢"
            
            # Product name
            product_name = "N/A"
            if 'Product Name' in st.session_state.data_clean.columns:
                names = df_sku['Product Name'].unique()
                if len(names) > 0:
                    product_name = str(names[0])[:30] + '...' if len(str(names[0])) > 30 else str(names[0])
            
            summary_data.append({
                'SKU': sku,
                'Product': product_name,
                'Forecast': int(last_3_months),
                'Safety Stock': int(safety_stock),
                'Total Required': int(last_3_months + safety_stock),
                'Volatility': f"{status_color} {volatility} ({cv:.0f}%)"
            })
    
    # Buat DataFrame summary
    df_summary = pd.DataFrame(summary_data)
    
    # Tampilkan summary
    st.dataframe(
        df_summary,
        use_container_width=True,
        height=500,
        column_config={
            'Forecast': st.column_config.NumberColumn(format='%d'),
            'Safety Stock': st.column_config.NumberColumn(format='%d'),
            'Total Required': st.column_config.NumberColumn(format='%d')
        }
    )
    
    # Summary metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total SKU Analyzed", len(df_summary))
    with col2:
        total_forecast = df_summary['Forecast'].sum()
        st.metric("Total Forecast Demand", f"{total_forecast:,.0f}")
    with col3:
        total_safety = df_summary['Safety Stock'].sum()
        st.metric("Total Safety Stock", f"{total_safety:,.0f}")
    
    # Narrative summary
    st.subheader("📝 Executive Summary")
    
    total_high = len(df_summary[df_summary['Volatility'].str.contains('High')])
    total_medium = len(df_summary[df_summary['Volatility'].str.contains('Medium')])
    total_low = len(df_summary[df_summary['Volatility'].str.contains('Low')])
    
    narrative = f"""
    **📊 RINGKASAN EKSEKUTIF - S&OP MEETING**
    
    **Situasi Terkini:**
    - Dari {len(df_summary)} SKU yang dianalisis:
        - 🟢 **{total_low} SKU** dengan volatilitas rendah (Stabil)
        - 🟡 **{total_medium} SKU** dengan volatilitas menengah (Perlu Monitoring)
        - 🔴 **{total_high} SKU** dengan volatilitas tinggi (Risk Stockout/Excess)
    
    **Kebutuhan Stok Bulan Depan:**
    - Total forecast demand: **{total_forecast:,.0f} unit**
    - Total safety stock required: **{total_safety:,.0f} unit**
    - Total stok yang perlu disiapkan: **{total_forecast + total_safety:,.0f} unit**
    """
    
    st.markdown(f"<div class='insight-box'>{narrative}</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 6: EXPORT & REPORT
# =============================================================================

with tabs[5]:
    st.header("📤 Export & Reporting")
    
    st.markdown("""
    <div class='insight-box'>
        <h4>📥 Export Data untuk Presentasi</h4>
        <p>Download hasil analisis dalam format CSV</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 Generate Export CSV"):
        with st.spinner("Menyiapkan file export..."):
            
            # Buat file CSV
            if 'df_summary' in locals():
                csv = df_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download S&OP Summary",
                    data=csv,
                    file_name=f"SOP_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        S&OP Forecast Studio Enterprise v2.0 | 
        Dibangun untuk Demand & Supply Planner Senior | 
        Last run: {datetime.now().strftime('%d %b %Y %H:%M:%S')}
    </div>
    """,
    unsafe_allow_html=True
)
