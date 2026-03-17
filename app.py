import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LAZY LOADING - Import library hanya saat diperlukan (optimasi memory)
# =============================================================================
def import_visualization():
    """Import visualization libraries lazily"""
    global plt, sns, px, go
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

def import_ml_libraries():
    """Import ML libraries lazily"""
    global Prophet, XGBRegressor, RandomForestRegressor, mean_absolute_percentage_error
    from prophet import Prophet
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_percentage_error

def import_google():
    """Import Google Sheets libraries lazily"""
    global gspread, Credentials, client
    import gspread
    from google.oauth2.service_account import Credentials

def import_utils():
    """Import utility libraries"""
    global joblib
    import joblib

# =============================================================================
# KONFIGURASI HALAMAN
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
# FUNGSI BANTU (HELPER FUNCTIONS)
# =============================================================================

@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def load_data_from_gsheet():
    """
    Membaca data dari Google Sheet menggunakan service account
    """
    try:
        import_google()
        
        # Ambil credentials dari Streamlit secrets
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scope
        )
        
        gc = gspread.authorize(credentials)
        
        # Buka spreadsheet
        url = "https://docs.google.com/spreadsheets/d/1PuoII49N-IWOaNO8fSMYGwuvFf1T68_Kez30WN9q8Ds/edit?gid=857579960#gid=857579960"
        sh = gc.open_by_url(url)
        
        # Ambil semua worksheet yang relevan
        data_sources = {}
        
        # Baca Sales Data
        try:
            worksheet = sh.worksheet("Sales")
            data = worksheet.get_all_values()
            headers = data[0]
            rows = data[1:]
            data_sources['sales'] = pd.DataFrame(rows, columns=headers)
        except Exception as e:
            st.warning(f"Tidak bisa membaca sheet Sales: {str(e)}")
        
        # Baca Stock On Hand (jika ada)
        try:
            worksheet = sh.worksheet("Stock_Onhand")
            data = worksheet.get_all_values()
            headers = data[0]
            rows = data[1:]
            data_sources['stock'] = pd.DataFrame(rows, columns=headers)
        except:
            data_sources['stock'] = None
        
        # Baca Product Master
        try:
            worksheet = sh.worksheet("Product_Master")
            data = worksheet.get_all_values()
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
    if df_sales is None:
        return None
    
    # Identifikasi kolom bulan (format: "Jan 24", "Feb 24", dll)
    month_columns = [col for col in df_sales.columns if any(month in col for month in 
                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
    
    # Pastikan kolom identifier ada
    id_columns = ['SKU_ID', 'Product Name', 'Sub Brand', 'SKU Tier']
    available_ids = [col for col in id_columns if col in df_sales.columns]
    
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
    
    # Drop baris dengan Sales NaN
    df_long = df_long.dropna(subset=['Sales'])
    
    # Fungsi untuk konversi period ke date
    def parse_period(period):
        try:
            # Format: "Jan 24", "Feb 24", dll
            return pd.to_datetime(period, format='%b %y')
        except:
            return pd.NaT
    
    df_long['Date'] = df_long['Period'].apply(parse_period)
    
    # Drop baris dengan Date NaT
    df_long = df_long.dropna(subset=['Date'])
    
    # Tambah kolom bulan dan tahun untuk analisis
    df_long['Month'] = df_long['Date'].dt.month
    df_long['Year'] = df_long['Date'].dt.year
    df_long['MonthName'] = df_long['Date'].dt.strftime('%b')
    df_long['Quarter'] = df_long['Date'].dt.quarter
    
    # Sort by SKU and Date
    df_long = df_long.sort_values(['SKU_ID', 'Date'])
    
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
    seasonality_idx = (monthly_avg / overall_avg * 100).to_dict()
    
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
    import_ml_libraries()
    
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
    
    # Tambahkan efek holiday (untuk Indonesia)
    # Bisa ditambahkan custom holidays nanti
    
    model.fit(df_prophet)
    
    # Buat dataframe future
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    # Hitung akurasi (in-sample)
    predictions = forecast.iloc[:len(df_prophet)]['yhat']
    actuals = df_prophet['y'].values
    mape = mean_absolute_percentage_error(actuals, predictions) * 100
    
    return model, forecast, mape

def run_forecast_xgboost(df_sku, periods=6):
    """
    Forecasting dengan XGBoost (menggunakan fitur time series)
    """
    import_ml_libraries()
    
    # Buat fitur dari tanggal
    df_features = df_sku.copy()
    df_features['month'] = df_features['Date'].dt.month
    df_features['quarter'] = df_features['Date'].dt.quarter
    df_features['year'] = df_features['Date'].dt.year
    df_features['dayofyear'] = df_features['Date'].dt.dayofyear
    df_features['lag_1'] = df_features['Sales'].shift(1)
    df_features['lag_2'] = df_features['Sales'].shift(2)
    df_features['lag_3'] = df_features['Sales'].shift(3)
    df_features['rolling_mean_3'] = df_features['Sales'].rolling(window=3).mean()
    
    # Drop NA
    df_features = df_features.dropna()
    
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
    mape = mean_absolute_percentage_error(y, predictions) * 100
    
    # Generate future dates
    last_date = df_sku['Date'].max()
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
    
    # Buat prediksi untuk masa depan (sederhana - menggunakan lag terakhir)
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
        rolling_mean = np.mean(last_values[-3:]) if len(last_values) >= 3 else np.mean(last_values)
        
        # Buat array fitur
        features = np.array([[future_month, future_quarter, future_year, 
                             lag_1, lag_2, lag_3, rolling_mean]])
        
        # Prediksi
        pred = model.predict(features)[0]
        future_predictions.append(pred)
        
        # Update last_values untuk iterasi berikutnya
        last_values = np.append(last_values[1:], pred)
    
    # Buat dataframe forecast
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': future_predictions,
        'yhat_lower': [p * 0.9 for p in future_predictions],  # Simple CI
        'yhat_upper': [p * 1.1 for p in future_predictions]
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
    
    return safety_stock

def export_to_excel(summary_df, forecast_df, filename):
    """
    Export hasil ke Excel dengan multiple sheets
    """
    import_utils()
    
    output = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Write each dataframe to different sheets
    summary_df.to_excel(output, sheet_name='S&OP Summary', index=False)
    forecast_df.to_excel(output, sheet_name='Forecast Details', index=False)
    
    # Close the writer
    output.close()
    
    return filename

def create_presentation_charts(df_sku, forecast_dict):
    """
    Membuat charts untuk presentasi (matplotlib style)
    """
    import_visualization()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('S&OP Forecast Analysis', fontsize=16, fontweight='bold')
    
    # Chart 1: Historical Trend
    axes[0, 0].plot(df_sku['Date'], df_sku['Sales'], 'b-', linewidth=2)
    axes[0, 0].set_title('Historical Sales Trend')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Chart 2: Forecast Comparison
    if forecast_dict:
        axes[0, 1].plot(forecast_dict['dates'], forecast_dict['prophet'], 
                       'r--', label='Prophet', linewidth=2)
        axes[0, 1].plot(forecast_dict['dates'], forecast_dict['xgboost'], 
                       'g--', label='XGBoost', linewidth=2)
        axes[0, 1].plot(forecast_dict['dates'], forecast_dict['ensemble'], 
                       'b-', label='Ensemble', linewidth=3)
        axes[0, 1].set_title('Forecast Comparison')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Forecast')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Chart 3: Seasonal Pattern
    monthly_avg = df_sku.groupby('Month')['Sales'].mean()
    axes[1, 0].bar(monthly_avg.index, monthly_avg.values, color='purple', alpha=0.7)
    axes[1, 0].set_title('Monthly Seasonality')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Sales')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Chart 4: YoY Growth
    yearly = df_sku.groupby('Year')['Sales'].sum()
    axes[1, 1].plot(yearly.index, yearly.values, 'go-', linewidth=2, markersize=8)
    axes[1, 1].set_title('Yearly Sales')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Total Sales')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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
    if st.session_state.data_clean is not None:
        st.success("✅ Terhubung ke Google Sheets")
        st.caption(f"Terakhir: {datetime.now().strftime('%d %b %Y %H:%M')}")
        
        # Info data
        total_sku = st.session_state.data_clean['SKU_ID'].nunique()
        total_records = len(st.session_state.data_clean)
        st.info(f"📊 {total_sku} SKU | {total_records} records")
    else:
        st.warning("⏳ Menunggu koneksi...")
    
    st.markdown("---")
    
    # Tombol load data
    if st.button("🔄 Refresh & Load Data", use_container_width=True, type="primary"):
        with st.spinner("Membaca data dari Google Sheets..."):
            st.session_state.data_sources = load_data_from_gsheet()
            if st.session_state.data_sources and 'sales' in st.session_state.data_sources:
                st.session_state.data_clean = clean_and_transform_data(
                    st.session_state.data_sources['sales']
                )
                st.rerun()
    
    st.markdown("---")
    
    # Filter jika data sudah ada
    if st.session_state.data_clean is not None:
        st.subheader("🔍 Filter Data")
        
        # Filter by Sub Brand
        brands = ['Semua Brand'] + sorted(st.session_state.data_clean['Sub Brand'].dropna().unique().tolist())
        selected_brand = st.selectbox("Sub Brand", brands)
        
        # Filter by SKU Tier
        tiers = ['Semua Tier'] + sorted(st.session_state.data_clean['SKU Tier'].dropna().unique().tolist())
        selected_tier = st.selectbox("SKU Tier", tiers)
        
        # Filter SKU berdasarkan brand dan tier
        df_filtered = st.session_state.data_clean.copy()
        if selected_brand != 'Semua Brand':
            df_filtered = df_filtered[df_filtered['Sub Brand'] == selected_brand]
        if selected_tier != 'Semua Tier':
            df_filtered = df_filtered[df_filtered['SKU Tier'] == selected_tier]
        
        sku_options = ['Semua SKU'] + sorted(df_filtered['SKU_ID'].unique().tolist())
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
# MAIN CONTENT - TABS
# =============================================================================

st.markdown("<h1 class='main-header'>📊 S&OP Forecast Studio Enterprise</h1>", unsafe_allow_html=True)

if st.session_state.data_clean is None:
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
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total SKU",
            st.session_state.data_clean['SKU_ID'].nunique(),
            help="Jumlah SKU unik"
        )
    with col2:
        st.metric(
            "Total Sub Brand",
            st.session_state.data_clean['Sub Brand'].nunique(),
            help="Jumlah Sub Brand"
        )
    with col3:
        date_range = f"{st.session_state.data_clean['Date'].min().strftime('%b %y')} - {st.session_state.data_clean['Date'].max().strftime('%b %y')}"
        st.metric("Periode Data", date_range)
    with col4:
        total_sales = st.session_state.data_clean['Sales'].sum()
        st.metric("Total Sales", f"{total_sales:,.0f}")
    
    st.markdown("---")
    
    # Data preview dengan opsi tampilan
    view_option = st.radio(
        "Tampilan Data:",
        ["Long Format (Time Series)", "Wide Format (Pivot Table)"],
        horizontal=True
    )
    
    if view_option == "Long Format (Time Series)":
        st.dataframe(
            st.session_state.data_clean[['SKU_ID', 'Product Name', 'Sub Brand', 'Date', 'Sales']],
            use_container_width=True,
            height=400
        )
    else:
        # Pivot untuk tampilan wide
        pivot_data = st.session_state.data_clean.pivot_table(
            index=['SKU_ID', 'Product Name', 'Sub Brand', 'SKU Tier'],
            columns='Date',
            values='Sales',
            aggfunc='sum'
        ).round(0)
        
        # Format kolom tanggal
        pivot_data.columns = [col.strftime('%b %y') for col in pivot_data.columns]
        
        st.dataframe(
            pivot_data,
            use_container_width=True,
            height=400
        )
    
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
        eda_sku = st.selectbox(
            "Pilih SKU untuk Analisis Detail",
            options=['Semua SKU (Aggregated)'] + sorted(st.session_state.data_clean['SKU_ID'].unique().tolist())
        )
    with col2:
        chart_type = st.selectbox(
            "Tipe Chart",
            ['Line Chart', 'Bar Chart', 'Area Chart', 'Box Plot']
        )
    
    # Filter data
    if eda_sku != 'Semua SKU (Aggregated)':
        df_eda = st.session_state.data_clean[st.session_state.data_clean['SKU_ID'] == eda_sku].copy()
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
            st.metric(
                "Sales L12M", 
                f"{kpis['total_sales_l12m']:,.0f}",
                help="Total penjualan 12 bulan terakhir"
            )
        with col2:
            st.metric(
                "Rata-rata Bulanan", 
                f"{kpis['avg_monthly_sales']:,.0f}"
            )
        with col3:
            st.metric(
                "Volatilitas (CV)", 
                f"{kpis['cv']:.1f}%",
                help=">50% = Highly Volatile",
                delta_color="inverse"
            )
        with col4:
            if kpis['yoy_growth'] is not None:
                st.metric(
                    "YoY Growth", 
                    f"{kpis['yoy_growth']:.1f}%",
                    delta=f"{kpis['yoy_growth']:.1f}%"
                )
            else:
                st.metric("YoY Growth", "N/A")
    
    st.markdown("---")
    
    # Time Series Plot
    st.subheader(f"📅 {title_prefix}")
    
    import_visualization()
    
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
        
        if kpis and 'seasonality_idx' in kpis:
            season_df = pd.DataFrame(
                list(kpis['seasonality_idx'].items()),
                columns=['Month', 'Index']
            )
            season_df['MonthName'] = season_df['Month'].apply(lambda x: calendar.month_abbr[x])
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
    
    # Statistik Detail
    with st.expander("📈 Lihat Statistik Detail"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Deskriptif Statistik")
            st.dataframe(df_eda['Sales'].describe().round(0).to_frame())
        
        with col2:
            st.subheader("Top Performing Months")
            top_months = df_eda.nlargest(5, 'Sales')[['Date', 'Sales']]
            top_months['Date'] = top_months['Date'].dt.strftime('%b %Y')
            top_months = top_months.rename(columns={'Date': 'Bulan', 'Sales': 'Penjualan'})
            st.dataframe(top_months)

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
        
        # Tampilkan info SKU
        sku_info = st.session_state.data_sources['sales'][
            st.session_state.data_sources['sales']['SKU_ID'] == st.session_state.selected_sku
        ].iloc[0] if 'sales' in st.session_state.data_sources else {}
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**SKU:** {st.session_state.selected_sku}")
        with col2:
            st.info(f"**Product:** {sku_info.get('Product Name', 'N/A')}")
        with col3:
            st.info(f"**Sub Brand:** {sku_info.get('Sub Brand', 'N/A')}")
        
        st.markdown("---")
        
        # Cek apakah user sudah klik run forecast
        if 'run_forecast' in locals() and run_forecast and st.session_state.selected_sku:
            with st.spinner("Menjalankan multiple forecasting models..."):
                
                results = {}
                metrics = {}
                
                # Prophet Forecast
                if 'Prophet' in forecast_model:
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
                
                # XGBoost Forecast
                if 'XGBoost' in forecast_model:
                    model_x, forecast_x, mape_x = run_forecast_xgboost(
                        df_sku,
                        periods=forecast_periods
                    )
                    results['xgboost'] = {
                        'model': model_x,
                        'forecast': forecast_x,
                        'mape': mape_x
                    }
                    metrics['XGBoost'] = mape_x
                
                # Ensemble Forecast
                if 'Ensemble' in forecast_model and 'prophet' in results and 'xgboost' in results:
                    ensemble = create_ensemble_forecast(
                        results['prophet']['forecast'],
                        results['xgboost']['forecast']
                    )
                    results['ensemble'] = {
                        'forecast': ensemble,
                        'mape': np.mean([mape_p, mape_x])
                    }
                    metrics['Ensemble'] = np.mean([mape_p, mape_x])
                
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
                
                st.success("✅ Forecasting selesai! Lihat hasil di bawah.")
        
        # Tampilkan hasil forecast jika ada
        if st.session_state.selected_sku in st.session_state.forecast_results:
            result = st.session_state.forecast_results[st.session_state.selected_sku]
            
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
                best_model = metrics_df.loc[metrics_df['MAPE (%)'].idxmin()]
                st.success(f"🏆 Best Model: **{best_model['Model']}** dengan MAPE {best_model['MAPE (%)']:.1f}%")
            
            with col2:
                # Bar chart perbandingan
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
            
            # Detailed forecast tables
            st.subheader("📊 Detailed Forecast by Model")
            
            tabs_forecast = st.tabs(list(names.values()) + ['Summary'])
            
            for i, (model_key, model_name) in enumerate(names.items()):
                if model_key in result['results']:
                    with tabs_forecast[i]:
                        forecast = result['results'][model_key]['forecast']
                        
                        # Format for display
                        display_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods).copy()
                        display_df['ds'] = display_df['ds'].dt.strftime('%b %Y')
                        display_df.columns = ['Periode', 'Forecast', 'Lower Bound', 'Upper Bound']
                        display_df = display_df.round(0)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(display_df, use_container_width=True)
                        with col2:
                            st.metric("MAPE", f"{result['results'][model_key]['mape']:.1f}%")
                            st.metric("Total Forecast", f"{display_df['Forecast'].sum():,.0f}")
            
            with tabs_forecast[-1]:
                # Summary table - ensemble jika ada, otherwise average
                if 'ensemble' in result['results']:
                    summary_forecast = result['results']['ensemble']['forecast']
                else:
                    # Average of available models
                    forecasts = []
                    for model_key in result['results']:
                        if model_key != 'ensemble':
                            forecasts.append(result['results'][model_key]['forecast']['yhat'].values)
                    
                    avg_forecast = np.mean(forecasts, axis=0)
                    summary_forecast = result['results'][list(result['results'].keys())[0]]['forecast'].copy()
                    summary_forecast['yhat'] = avg_forecast
                
                display_summary = summary_forecast[['ds', 'yhat']].tail(forecast_periods).copy()
                display_summary['ds'] = display_summary['ds'].dt.strftime('%b %Y')
                display_summary.columns = ['Periode', 'Recommended Forecast']
                
                st.dataframe(display_summary, use_container_width=True)
                
                # Safety stock recommendation
                safety_stock = calculate_safety_stock(
                    result['historical'],
                    display_summary['Recommended Forecast'].iloc[0],
                    service_level=confidence_level
                )
                
                st.info(f"""
                **📦 Rekomendasi Safety Stock:**
                - Untuk service level {int(confidence_level*100)}%, safety stock yang disarankan: **{safety_stock:,.0f} unit**
                - Total stok yang perlu disiapkan bulan depan: **{display_summary['Recommended Forecast'].iloc[0] + safety_stock:,.0f} unit**
                """)

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
            row.update(result['metrics'])
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Tampilkan heatmap performa
        st.subheader("Model Performance Heatmap")
        
        # Prepare data for heatmap
        heatmap_data = df_comparison.set_index('SKU')
        
        fig_heatmap = px.imshow(
            heatmap_data.T,
            text_auto='.1f',
            aspect="auto",
            color_continuous_scale='RdYlGn_r',
            title='Model MAPE Comparison (% - Lower is Better)'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tabel perbandingan
        st.subheader("Detail Comparison")
        st.dataframe(df_comparison.round(2), use_container_width=True)
        
        # Best model per SKU
        st.subheader("🏆 Best Model per SKU")
        best_models = []
        for sku in df_comparison['SKU']:
            row = df_comparison[df_comparison['SKU'] == sku].iloc[0]
            metrics = {k: v for k, v in row.items() if k != 'SKU' and pd.notna(v)}
            if metrics:
                best_model = min(metrics, key=metrics.get)
                best_models.append({
                    'SKU': sku,
                    'Best Model': best_model,
                    'MAPE': metrics[best_model]
                })
        
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
    
    # Buat summary untuk semua SKU
    all_skus = st.session_state.data_clean['SKU_ID'].unique()
    
    summary_data = []
    
    with st.spinner("Menghitung rekomendasi untuk semua SKU..."):
        for sku in all_skus[:50]:  # Batasi 50 SKU untuk performa
            df_sku = st.session_state.data_clean[st.session_state.data_clean['SKU_ID'] == sku].copy()
            
            # Ambil info produk
            sku_info = st.session_state.data_sources['sales'][
                st.session_state.data_sources['sales']['SKU_ID'] == sku
            ].iloc[0] if 'sales' in st.session_state.data_sources else {}
            
            # Simple forecast (average of last 3 months)
            last_3_months = df_sku.sort_values('Date').tail(3)['Sales'].mean()
            
            # Safety stock
            safety_stock = calculate_safety_stock(df_sku, last_3_months, service_level=0.95)
            
            # Hitung CV untuk klasifikasi
            cv = (df_sku['Sales'].std() / df_sku['Sales'].mean()) * 100
            
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
            
            summary_data.append({
                'SKU': sku,
                'Product': str(sku_info.get('Product Name', 'N/A'))[:30] + '...' if len(str(sku_info.get('Product Name', ''))) > 30 else sku_info.get('Product Name', 'N/A'),
                'Sub Brand': sku_info.get('Sub Brand', 'N/A'),
                'Tier': sku_info.get('SKU Tier', 'N/A'),
                'Forecast (Next Month)': int(last_3_months),
                'Safety Stock': int(safety_stock),
                'Total Required': int(last_3_months + safety_stock),
                'Volatility': f"{status_color} {volatility} ({cv:.0f}%)",
                'YoY Growth (%)': int(calculate_kpis(df_sku)['yoy_growth']) if calculate_kpis(df_sku) and calculate_kpis(df_sku)['yoy_growth'] else 0
            })
    
    # Buat DataFrame summary
    df_summary = pd.DataFrame(summary_data)
    
    # Filter and sort options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_tier = st.multiselect(
            "Filter SKU Tier",
            options=df_summary['Tier'].unique(),
            default=df_summary['Tier'].unique()
        )
    with col2:
        filter_volatility = st.multiselect(
            "Filter Volatility",
            options=['Low', 'Medium', 'High'],
            default=['Low', 'Medium', 'High']
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ['Forecast (Next Month)', 'Safety Stock', 'YoY Growth (%)']
        )
    
    # Apply filters
    mask_volatility = df_summary['Volatility'].str.contains('|'.join(filter_volatility))
    mask_tier = df_summary['Tier'].isin(filter_tier)
    df_filtered = df_summary[mask_volatility & mask_tier].sort_values(sort_by, ascending=False)
    
    # Tampilkan summary
    st.dataframe(
        df_filtered,
        use_container_width=True,
        height=500,
        column_config={
            'Volatility': st.column_config.Column(
                'Volatility',
                help='Volatilitas penjualan (CV)'
            ),
            'Forecast (Next Month)': st.column_config.NumberColumn(
                'Forecast',
                format='%d'
            ),
            'Safety Stock': st.column_config.NumberColumn(
                'Safety Stock',
                format='%d'
            ),
            'Total Required': st.column_config.NumberColumn(
                'Total Required',
                format='%d'
            )
        }
    )
    
    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total SKU Analyzed", len(df_filtered))
    with col2:
        total_forecast = df_filtered['Forecast (Next Month)'].sum()
        st.metric("Total Forecast Demand", f"{total_forecast:,.0f}")
    with col3:
        total_safety = df_filtered['Safety Stock'].sum()
        st.metric("Total Safety Stock", f"{total_safety:,.0f}")
    with col4:
        high_volatility = len(df_filtered[df_filtered['Volatility'].str.contains('High')])
        st.metric("High Volatility SKUs", high_volatility)
    
    st.markdown("---")
    
    # Narrative summary
    st.subheader("📝 Executive Summary")
    
    total_high = len(df_filtered[df_filtered['Volatility'].str.contains('High')])
    total_medium = len(df_filtered[df_filtered['Volatility'].str.contains('Medium')])
    total_low = len(df_filtered[df_filtered['Volatility'].str.contains('Low')])
    
    narrative = f"""
    **📊 RINGKASAN EKSEKUTIF - S&OP MEETING**
    
    **Situasi Terkini:**
    - Dari {len(df_filtered)} SKU yang dianalisis:
        - 🟢 **{total_low} SKU** dengan volatilitas rendah (Stabil)
        - 🟡 **{total_medium} SKU** dengan volatilitas menengah (Perlu Monitoring)
        - 🔴 **{total_high} SKU** dengan volatilitas tinggi (Risk Stockout/Excess)
    
    **Kebutuhan Stok Bulan Depan:**
    - Total forecast demand: **{total_forecast:,.0f} unit**
    - Total safety stock required: **{total_safety:,.0f} unit**
    - Total stok yang perlu disiapkan: **{total_forecast + total_safety:,.0f} unit**
    
    **Rekomendasi Tindakan:**
    1. **Prioritas Tinggi**: Fokus pada {total_high} SKU dengan volatilitas tinggi - review mingguan
    2. **Prioritas Menengah**: Monitor {total_medium} SKU dengan volatilitas menengah - review bulanan
    3. **Prioritas Rendah**: Maintain {total_low} SKU dengan volatilitas rendah - review kuartalan
    
    **Catatan untuk Meeting:**
    - Data forecast menggunakan ensemble model (Prophet + XGBoost)
    - Safety stock dihitung dengan service level 95%
    - Perlu validasi dengan tim Sales untuk promo yang direncanakan
    """
    
    st.markdown(f"<div class='insight-box'>{narrative}</div>", unsafe_allow_html=True)
    
    # Editable text area untuk notes meeting
    meeting_notes = st.text_area("📝 Meeting Notes:", height=150)

# =============================================================================
# TAB 6: EXPORT & REPORT
# =============================================================================

with tabs[5]:
    st.header("📤 Export & Reporting")
    
    st.markdown("""
    <div class='insight-box'>
        <h4>📥 Export Data untuk Presentasi</h4>
        <p>Download hasil analisis dalam berbagai format untuk keperluan meeting</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Options")
        
        export_format = st.radio(
            "Pilih Format Export:",
            ["Excel (Multiple Sheets)", "CSV", "PowerPoint Charts"]
        )
        
        export_scope = st.radio(
            "Scope Data:",
            ["Semua SKU", "SKU Terpilih Saja"]
        )
        
        if st.button("📥 Generate Export", type="primary"):
            with st.spinner("Menyiapkan file export..."):
                
                if export_format == "Excel (Multiple Sheets)":
                    # Buat Excel dengan multiple sheets
                    output_file = f"SOP_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    
                    # Siapkan data untuk export
                    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                        # Sheet 1: Summary
                        if 'df_filtered' in locals():
                            df_filtered.to_excel(writer, sheet_name='S&OP Summary', index=False)
                        
                        # Sheet 2: Forecast Results
                        if st.session_state.forecast_results:
                            forecast_rows = []
                            for sku, result in st.session_state.forecast_results.items():
                                if 'results' in result and 'ensemble' in result['results']:
                                    forecast = result['results']['ensemble']['forecast'].tail(6)
                                    for _, row in forecast.iterrows():
                                        forecast_rows.append({
                                            'SKU': sku,
                                            'Periode': row['ds'].strftime('%b %Y'),
                                            'Forecast': row['yhat'],
                                            'Lower_Bound': row['yhat_lower'],
                                            'Upper_Bound': row['yhat_upper']
                                        })
                            
                            if forecast_rows:
                                pd.DataFrame(forecast_rows).to_excel(
                                    writer, sheet_name='Forecast Details', index=False
                                )
                        
                        # Sheet 3: Model Performance
                        if st.session_state.forecast_results:
                            perf_rows = []
                            for sku, result in st.session_state.forecast_results.items():
                                if 'metrics' in result:
                                    for model, mape in result['metrics'].items():
                                        perf_rows.append({
                                            'SKU': sku,
                                            'Model': model,
                                            'MAPE': mape
                                        })
                            
                            if perf_rows:
                                pd.DataFrame(perf_rows).to_excel(
                                    writer, sheet_name='Model Performance', index=False
                                )
                    
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel Report",
                            data=f,
                            file_name=output_file,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                elif export_format == "PowerPoint Charts":
                    st.info("Fitur PowerPoint Charts akan segera hadir!")
                
                else:  # CSV
                    if 'df_filtered' in locals():
                        csv = df_filtered.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"SOP_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
    
    with col2:
        st.subheader("📈 Preview Report")
        
        if 'df_filtered' in locals():
            st.dataframe(df_filtered.head(10), use_container_width=True)
            
            # Simple chart untuk report
            fig_report = px.bar(
                df_filtered.head(15),
                x='SKU',
                y='Forecast (Next Month)',
                color='Volatility',
                title='Top 15 SKU by Forecast Demand'
            )
            st.plotly_chart(fig_report, use_container_width=True)

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
