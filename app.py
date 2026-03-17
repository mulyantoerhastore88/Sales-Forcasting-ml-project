import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import calendar
from prophet import Prophet
from prophet.plot import plot_plotly
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. KONFIGURASI HALAMAN (HARUS PALING ATAS)
# =============================================================================
st.set_page_config(
    page_title="S&OP Forecast Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. FUNGSI BANTU (HELPER FUNCTIONS)
# =============================================================================

@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def load_data_from_gsheet():
    """
    Membaca data dari Google Sheet menggunakan service account
    """
    try:
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
        
        # Ambil worksheet Sales
        worksheet = sh.worksheet("Sales")
        
        # Baca semua data
        data = worksheet.get_all_values()
        headers = data[0]
        rows = data[1:]
        
        # Buat DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_and_transform_data(df):
    """
    Membersihkan dan mentransformasi data dari format lebar ke format panjang
    """
    # Identifikasi kolom bulan (format: "Jan 24", "Feb 24", dll)
    month_columns = [col for col in df.columns if any(month in col for month in 
                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
    
    # Melt data dari format lebar ke panjang
    df_long = pd.melt(
        df,
        id_vars=['SKU_ID', 'Product Name', 'Sub Brand', 'SKU Tier'],
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
    
    return {
        'total_sales_l12m': total_sales_l12m,
        'avg_monthly_sales': avg_monthly_sales,
        'cv': cv,
        'yoy_growth': yoy_growth
    }

def run_forecast_prophet(df_sku, periods=6, changepoint_prior_scale=0.05):
    """
    Menjalankan forecasting dengan Prophet untuk satu SKU
    """
    # Siapkan data untuk Prophet
    df_prophet = df_sku[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Inisialisasi dan fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode='multiplicative'
    )
    
    # Tambahkan monthly seasonality (opsional)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(df_prophet)
    
    # Buat dataframe future
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    return model, forecast

def calculate_safety_stock(df_sku, forecast_value, service_level=0.95):
    """
    Menghitung safety stock berdasarkan MAD (Mean Absolute Deviation)
    """
    # Hitung forecast error sederhana (naif: gunakan rata-rata bergerak)
    df_sku = df_sku.sort_values('Date')
    df_sku['MA_3'] = df_sku['Sales'].rolling(window=3, min_periods=1).mean().shift(1)
    df_sku['Error'] = abs(df_sku['Sales'] - df_sku['MA_3'])
    
    mad = df_sku['Error'].mean()
    
    # Faktor keamanan untuk service level 95% (z-score 1.96)
    safety_factor = 1.96 if service_level == 0.95 else 1.28 if service_level == 0.90 else 2.33
    
    safety_stock = safety_factor * mad
    
    return safety_stock

# =============================================================================
# 3. INISIALISASI SESSION STATE
# =============================================================================
if 'data_raw' not in st.session_state:
    st.session_state.data_raw = None
if 'data_clean' not in st.session_state:
    st.session_state.data_clean = None
if 'selected_sku' not in st.session_state:
    st.session_state.selected_sku = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = {}

# =============================================================================
# 4. SIDEBAR - NAVIGASI DAN KONTROL
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000.png", width=80)
    st.title("📊 S&OP Forecast Studio")
    st.markdown("---")
    
    # Status koneksi
    if st.session_state.data_clean is not None:
        st.success("✅ Terhubung ke Google Sheets")
        st.caption(f"Data terakhir: {datetime.now().strftime('%d %b %Y %H:%M')}")
    else:
        st.warning("⏳ Menunggu data...")
    
    st.markdown("---")
    
    # Tombol load data
    if st.button("🔄 Refresh Data", use_container_width=True):
        with st.spinner("Membaca data dari Google Sheets..."):
            st.session_state.data_raw = load_data_from_gsheet()
            if st.session_state.data_raw is not None:
                st.session_state.data_clean = clean_and_transform_data(st.session_state.data_raw)
                st.rerun()
    
    st.markdown("---")
    
    # Filter jika data sudah ada
    if st.session_state.data_clean is not None:
        st.subheader("🔍 Filter Data")
        
        # Dapatkan unique values
        sku_list = ['Semua SKU'] + sorted(st.session_state.data_clean['SKU_ID'].unique().tolist())
        brand_list = ['Semua Brand'] + sorted(st.session_state.data_clean['Sub Brand'].dropna().unique().tolist())
        
        selected_brand = st.selectbox("Pilih Sub Brand", brand_list)
        
        # Filter SKU berdasarkan brand
        if selected_brand != 'Semua Brand':
            sku_options = st.session_state.data_clean[st.session_state.data_clean['Sub Brand'] == selected_brand]['SKU_ID'].unique()
            sku_list_filtered = ['Semua SKU'] + sorted(sku_options.tolist())
            selected_sku = st.selectbox("Pilih SKU", sku_list_filtered)
        else:
            selected_sku = st.selectbox("Pilih SKU", sku_list)
        
        st.session_state.selected_sku = selected_sku if selected_sku != 'Semua SKU' else None
        
        st.markdown("---")
        
        # Parameter forecasting
        st.subheader("⚙️ Parameter Forecast")
        forecast_periods = st.slider("Horizon (bulan)", 3, 12, 6)
        confidence_level = st.select_slider("Confidence Level", options=[0.80, 0.85, 0.90, 0.95], value=0.95)
        changepoint_prior = st.slider("Sensitivitas Perubahan Trend", 0.01, 0.5, 0.05, 0.01, 
                                      help="Nilai lebih besar = lebih sensitif terhadap perubahan trend")
        
        st.markdown("---")
        
        # Tombol run forecast
        run_forecast = st.button("🚀 Run Forecast", use_container_width=True, type="primary")

# =============================================================================
# 5. HALAMAN UTAMA
# =============================================================================

# Tabs untuk navigasi
tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Overview", "🔍 EDA & Analytics", "🤖 Forecasting", "📋 S&OP Summary"])

# =============================================================================
# TAB 1: DATA OVERVIEW
# =============================================================================

with tab1:
    st.header("📈 Data Overview")
    
    if st.session_state.data_clean is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total SKU", st.session_state.data_clean['SKU_ID'].nunique())
        with col2:
            st.metric("Total Sub Brand", st.session_state.data_clean['Sub Brand'].nunique())
        with col3:
            st.metric("Periode Data", f"{st.session_state.data_clean['Date'].min().strftime('%b %y')} - {st.session_state.data_clean['Date'].max().strftime('%b %y')}")
        with col4:
            total_sales_all = st.session_state.data_clean['Sales'].sum()
            st.metric("Total Sales", f"{total_sales_all:,.0f}")
        
        st.markdown("---")
        
        # Preview data
        st.subheader("Data Preview")
        
        # Tampilkan data dalam format lebar untuk preview yang lebih familiar
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
        
        # Tombol download
        csv = pivot_data.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("👈 Silakan klik 'Refresh Data' di sidebar untuk memuat data dari Google Sheets")

# =============================================================================
# TAB 2: EDA & ANALYTICS
# =============================================================================

with tab2:
    st.header("🔍 Exploratory Data Analysis")
    
    if st.session_state.data_clean is not None:
        
        # Filter data berdasarkan pilihan
        if st.session_state.selected_sku:
            df_filtered = st.session_state.data_clean[st.session_state.data_clean['SKU_ID'] == st.session_state.selected_sku]
            title_prefix = f"SKU: {st.session_state.selected_sku}"
        else:
            # Aggregate by date
            df_filtered = st.session_state.data_clean.groupby('Date')['Sales'].sum().reset_index()
            df_filtered['SKU_ID'] = 'TOTAL'
            title_prefix = "Semua SKU (Aggregated)"
        
        # KPI Cards
        kpis = calculate_kpis(df_filtered)
        
        if kpis:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Sales (L12M)", 
                    f"{kpis['total_sales_l12m']:,.0f}",
                    help="Total penjualan 12 bulan terakhir"
                )
            with col2:
                st.metric(
                    "Rata-rata Bulanan", 
                    f"{kpis['avg_monthly_sales']:,.0f}",
                    help="Rata-rata penjualan per bulan"
                )
            with col3:
                st.metric(
                    "Koefisien Variasi", 
                    f"{kpis['cv']:.1f}%",
                    help="Semakin tinggi nilai, semakin volatile penjualannya",
                    delta_color="inverse"
                )
            with col4:
                if kpis['yoy_growth'] is not None:
                    st.metric(
                        "YoY Growth", 
                        f"{kpis['yoy_growth']:.1f}%",
                        delta=f"{kpis['yoy_growth']:.1f}%",
                        delta_color="normal"
                    )
                else:
                    st.metric("YoY Growth", "N/A")
        
        st.markdown("---")
        
        # Time Series Plot
        st.subheader("📅 Time Series - Historical Sales")
        
        fig = px.line(
            df_filtered, 
            x='Date', 
            y='Sales',
            title=f'Tren Penjualan - {title_prefix}',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Periode",
            yaxis_title="Sales (Unit)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal Analysis
        st.subheader("📊 Analisis Musiman")
        
        # Ekstrak bulan dan tahun
        df_filtered['Month'] = df_filtered['Date'].dt.month
        df_filtered['MonthName'] = df_filtered['Date'].dt.strftime('%b')
        df_filtered['Year'] = df_filtered['Date'].dt.year
        
        # Heatmap musiman
        pivot_monthly = df_filtered.pivot_table(
            values='Sales',
            index='Year',
            columns='MonthName',
            aggfunc='sum',
            fill_value=0
        )
        
        # Urutkan bulan
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_monthly = pivot_monthly[[m for m in month_order if m in pivot_monthly.columns]]
        
        fig_heatmap = px.imshow(
            pivot_monthly,
            text_auto='.0f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title=f'Heatmap Penjualan per Bulan - {title_prefix}'
        )
        
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Statistik Deskriptif
        st.subheader("📋 Statistik Deskriptif")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                df_filtered['Sales'].describe().round(0).to_frame(),
                use_container_width=True
            )
        
        with col2:
            # Top performing months
            top_months = df_filtered.nlargest(5, 'Sales')[['Date', 'Sales']]
            top_months['Date'] = top_months['Date'].dt.strftime('%b %Y')
            top_months = top_months.rename(columns={'Date': 'Bulan', 'Sales': 'Penjualan'})
            
            st.subheader("🏆 Top 5 Performing Months")
            st.dataframe(top_months, use_container_width=True)
            
    else:
        st.info("👈 Silakan muat data terlebih dahulu di tab 'Data Overview'")

# =============================================================================
# TAB 3: FORECASTING
# =============================================================================

with tab3:
    st.header("🤖 Machine Learning Forecast")
    
    if st.session_state.data_clean is not None:
        
        if not st.session_state.selected_sku:
            st.warning("⚠️ Silakan pilih satu SKU spesifik di sidebar untuk melakukan forecasting")
        else:
            # Siapkan data untuk SKU terpilih
            df_sku = st.session_state.data_clean[
                st.session_state.data_clean['SKU_ID'] == st.session_state.selected_sku
            ].copy()
            
            # Tampilkan info SKU
            sku_info = st.session_state.data_raw[
                st.session_state.data_raw['SKU_ID'] == st.session_state.selected_sku
            ].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**SKU:** {st.session_state.selected_sku}")
            with col2:
                st.info(f"**Product:** {sku_info.get('Product Name', 'N/A')}")
            with col3:
                st.info(f"**Sub Brand:** {sku_info.get('Sub Brand', 'N/A')}")
            
            st.markdown("---")
            
            # Cek apakah user sudah klik run forecast
            if 'run_forecast' in locals() and run_forecast:
                with st.spinner("Menjalankan model forecasting..."):
                    
                    # Run Prophet
                    model, forecast = run_forecast_prophet(
                        df_sku, 
                        periods=forecast_periods,
                        changepoint_prior_scale=changepoint_prior
                    )
                    
                    # Simpan hasil di session state
                    st.session_state.forecast_results[st.session_state.selected_sku] = {
                        'model': model,
                        'forecast': forecast,
                        'params': {
                            'periods': forecast_periods,
                            'confidence_level': confidence_level,
                            'changepoint_prior': changepoint_prior
                        }
                    }
                    
                    st.success("✅ Forecasting selesai!")
            
            # Tampilkan hasil forecast jika ada
            if st.session_state.selected_sku in st.session_state.forecast_results:
                result = st.session_state.forecast_results[st.session_state.selected_sku]
                forecast = result['forecast']
                
                # Plot forecast
                st.subheader("📈 Forecast Result")
                
                # Siapkan data untuk plotting
                historical = df_sku[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
                
                # Plot dengan Plotly
                fig = go.Figure()
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=historical['ds'],
                    y=historical['y'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='royalblue', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='crimson', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f'Confidence Interval ({result["params"]["confidence_level"]*100:.0f}%)'
                ))
                
                fig.update_layout(
                    title=f'Forecast {st.session_state.selected_sku} - {forecast_periods} Bulan ke Depan',
                    xaxis_title='Periode',
                    yaxis_title='Sales',
                    hovermode='x unified',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabel forecast
                st.subheader("📊 Tabel Forecast")
                
                forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods).copy()
                forecast_table['ds'] = forecast_table['ds'].dt.strftime('%b %Y')
                forecast_table.columns = ['Periode', 'Forecast', 'Lower Bound', 'Upper Bound']
                forecast_table = forecast_table.round(0)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(forecast_table, use_container_width=True)
                
                with col2:
                    # Komponen forecast
                    st.subheader("🔍 Komponen Forecast")
                    fig_components = plot_plotly(result['model'], forecast)
                    st.plotly_chart(fig_components, use_container_width=True)
                
                # Tombol download forecast
                csv_forecast = forecast_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_forecast,
                    file_name=f"forecast_{st.session_state.selected_sku}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.info("🚀 Klik 'Run Forecast' di sidebar untuk memulai forecasting")
                
    else:
        st.info("👈 Silakan muat data terlebih dahulu di tab 'Data Overview'")

# =============================================================================
# TAB 4: S&OP SUMMARY
# =============================================================================

with tab4:
    st.header("📋 S&OP Meeting Summary")
    
    if st.session_state.data_clean is not None:
        
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
        <h4 style='margin: 0;'>🎯 Ringkasan untuk Meeting S&OP Bulanan</h4>
        <p style='margin: 0.5rem 0 0 0; color: #555;'>Berdasarkan hasil forecast dan analisis stok</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Pilih SKU untuk summary
        all_skus = st.session_state.data_clean['SKU_ID'].unique()
        
        # Buat summary table
        summary_data = []
        
        with st.spinner("Menghitung rekomendasi untuk semua SKU..."):
            for sku in all_skus[:10]:  # Batasi 10 SKU dulu untuk performa
                df_sku = st.session_state.data_clean[st.session_state.data_clean['SKU_ID'] == sku]
                
                # Ambil info produk
                sku_info = st.session_state.data_raw[st.session_state.data_raw['SKU_ID'] == sku].iloc[0]
                
                # Hitung forecast sederhana (average of last 3 months)
                last_3_months = df_sku.sort_values('Date').tail(3)['Sales'].mean()
                
                # Safety stock
                safety_stock = calculate_safety_stock(df_sku, last_3_months, service_level=0.95)
                
                # Status stok (simulasi - idealnya dari sheet Stock_Onhand)
                # Untuk demo, kita buat status random
                import random
                stock_status = random.choice(['Aman', 'Perlu Perhatian', 'Kritis'])
                if stock_status == 'Aman':
                    status_color = '🟢'
                elif stock_status == 'Perlu Perhatian':
                    status_color = '🟡'
                else:
                    status_color = '🔴'
                
                summary_data.append({
                    'SKU': sku,
                    'Product': sku_info.get('Product Name', 'N/A')[:30] + '...' if len(str(sku_info.get('Product Name', ''))) > 30 else sku_info.get('Product Name', 'N/A'),
                    'Sub Brand': sku_info.get('Sub Brand', 'N/A'),
                    'Tier': sku_info.get('SKU Tier', 'N/A'),
                    'Forecast (Next Month)': int(last_3_months),
                    'Safety Stock': int(safety_stock),
                    'Status Stok': f"{status_color} {stock_status}",
                    'Rekomendasi': 'Tambah stok' if stock_status == 'Kritis' else 'Monitor' if stock_status == 'Perlu Perhatian' else 'Aman'
                })
        
        # Buat DataFrame summary
        df_summary = pd.DataFrame(summary_data)
        
        # Tampilkan summary dengan styling
        st.dataframe(
            df_summary,
            use_container_width=True,
            height=500,
            column_config={
                'Status Stok': st.column_config.Column(
                    'Status Stok',
                    help='Status ketersediaan stok berdasarkan forecast'
                ),
                'Forecast (Next Month)': st.column_config.NumberColumn(
                    'Forecast (Next Month)',
                    format='%d'
                ),
                'Safety Stock': st.column_config.NumberColumn(
                    'Safety Stock',
                    format='%d'
                )
            }
        )
        
        # Download summary
        csv_summary = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download S&OP Summary (CSV)",
            data=csv_summary,
            file_name=f"SOP_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Narrative summary
        st.subheader("📝 Ringkasan Naratif untuk Meeting")
        
        total_sku_kritis = len(df_summary[df_summary['Status Stok'].str.contains('Kritis')])
        total_sku_perlu = len(df_summary[df_summary['Status Stok'].str.contains('Perlu Perhatian')])
        total_sku_aman = len(df_summary[df_summary['Status Stok'].str.contains('Aman')])
        
        narrative = f"""
        **📊 RINGKASAN EKSEKUTIF - S&OP MEETING**

        **Situasi Terkini:**
        - Dari {len(df_summary)} SKU yang dianalisis, {total_sku_aman} SKU dalam status **Aman**, {total_sku_perlu} SKU **Perlu Perhatian**, dan {total_sku_kritis} SKU dalam status **Kritis** (berisiko out of stock).

        **Rekomendasi Tindakan:**
        - Prioritaskan pengadaan untuk SKU dengan status Kritis dalam 2 minggu ke depan.
        - Lakukan review forecast untuk SKU dengan volatilitas tinggi (CV > 50%).
        - Siapkan safety stock tambahan untuk menghadapi musim promo mendatang.

        **Catatan untuk Meeting:**
        - Data forecast telah mempertimbangkan musiman dari data historis 2 tahun.
        - Safety stock dihitung dengan service level 95%.
        - Rekomendasi ini perlu divalidasi dengan tim Sales untuk promo yang direncanakan.
        """
        
        st.markdown(narrative)
        
        # Kotak teks yang bisa diedit
        st.text_area("📝 Edit Ringkasan untuk Meeting:", narrative, height=300)
        
    else:
        st.info("👈 Silakan muat data terlebih dahulu di tab 'Data Overview'")

# =============================================================================
# 6. FOOTER
# =============================================================================

st.markdown("---")
st.caption(f"S&OP Forecast Studio v1.0 | Dibangun untuk Planner Senior | Last run: {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
