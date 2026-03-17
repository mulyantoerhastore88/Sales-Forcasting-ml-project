import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="SalesForecast AI - Inventory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helper Functions ----------
@st.cache_data(ttl=3600)  # Cache data selama 1 jam
def load_data_from_gsheet():
    """Load data from Google Sheets."""
    try:
        # Untuk deployment, gunakan secrets management Streamlit
        # Setup credentials
        scope = ["https://spreadsheets.google.com/feeds", 
                 "https://www.googleapis.com/auth/drive"]
        
        # Ambil credentials dari Streamlit secrets
        credentials = {
            "type": st.secrets["gcp"]["type"],
            "project_id": st.secrets["gcp"]["project_id"],
            "private_key_id": st.secrets["gcp"]["private_key_id"],
            "private_key": st.secrets["gcp"]["private_key"],
            "client_email": st.secrets["gcp"]["client_email"],
            "client_id": st.secrets["gcp"]["client_id"],
            "auth_uri": st.secrets["gcp"]["auth_uri"],
            "token_uri": st.secrets["gcp"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp"]["client_x509_cert_url"]
        }
        
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials, scope)
        client = gspread.authorize(creds)
        
        # Buka spreadsheet
        sheet = client.open_by_key('1PuoII49N-IWOaNO8fSMYGwuvFf1T68_Kez30WN9q8Ds')
        
        # Ambil data dari tab 'Sales'
        sales_worksheet = sheet.worksheet("Sales")
        data = sales_worksheet.get_all_values()
        headers = data[0]
        rows = data[1:]
        
        df = pd.DataFrame(rows, columns=headers)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Sheets: {e}")
        return None

@st.cache_data
def prepare_data_for_forecasting(df):
    """Transform data from wide to long format and create features."""
    if df is None:
        return None
    
    # Identifikasi kolom bulan (dari Jan 24 sampai Feb 26)
    month_cols = [col for col in df.columns if ' ' in col and any(month in col for month in 
                  ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])]
    
    # Ubah dari wide ke long format
    df_long = pd.melt(
        df,
        id_vars=['No', 'SKU GOA', 'SKU_ID', 'Product Name', 'Notes', 'Sub Brand', 'SKU Tier'],
        value_vars=month_cols,
        var_name='Month_Year',
        value_name='Sales'
    )
    
    # Konversi Sales ke numeric, handle error
    df_long['Sales'] = pd.to_numeric(df_long['Sales'], errors='coerce').fillna(0)
    
    # Parse Month_Year ke datetime
    def parse_month_year(mmyy):
        try:
            return datetime.strptime(mmyy, '%b %y')
        except:
            return None
    
    df_long['Date'] = df_long['Month_Year'].apply(parse_month_year)
    df_long = df_long.dropna(subset=['Date']).sort_values(['SKU_ID', 'Date'])
    
    # Feature Engineering
    df_long['Month'] = df_long['Date'].dt.month
    df_long['Year'] = df_long['Date'].dt.year
    df_long['Quarter'] = df_long['Date'].dt.quarter
    
    # Lag features (penjualan bulan sebelumnya)
    df_long['Lag_1'] = df_long.groupby('SKU_ID')['Sales'].shift(1)
    df_long['Lag_2'] = df_long.groupby('SKU_ID')['Sales'].shift(2)
    df_long['Lag_3'] = df_long.groupby('SKU_ID')['Sales'].shift(3)
    
    # Rolling means
    df_long['Rolling_Mean_3'] = df_long.groupby('SKU_ID')['Sales'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    
    # Hapus baris dengan NaN (data awal yang tidak punya lag)
    df_ml = df_long.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3']).reset_index(drop=True)
    
    return df_ml, df_long

@st.cache_resource
def load_model():
    """Load the trained model from file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
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
    
    # Jika tidak ada model, buat model sederhana untuk demo
    st.info("Model tidak ditemukan. Menggunakan model dummy untuk demo.")
    return "dummy"

def dummy_predict(sku_data, month, lag_1, lag_2, lag_3, rolling_mean):
    """Simple prediction for demo when no model exists."""
    # Weighted average of lags and rolling mean
    pred = (0.4 * lag_1 + 0.3 * lag_2 + 0.2 * lag_3 + 0.1 * rolling_mean)
    # Add seasonal factor based on month
    seasonal_factors = {1:1.1, 2:0.9, 3:1.0, 4:1.0, 5:1.1, 6:1.0, 
                        7:0.9, 8:0.9, 9:1.0, 10:1.1, 11:1.2, 12:1.3}
    pred = pred * seasonal_factors.get(month, 1.0)
    return max(0, round(pred, 0))

# ---------- Load Data ----------
with st.spinner('Memuat data dari Google Sheets...'):
    raw_df = load_data_from_gsheet()

if raw_df is not None:
    ml_df, full_df = prepare_data_for_forecasting(raw_df)
    model = load_model()
else:
    st.stop()

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center;'>🔍 Filter Data</h2>", unsafe_allow_html=True)
    
    # Pilih SKU
    sku_list = ml_df['SKU_ID'].unique()
    selected_sku = st.selectbox("Pilih SKU", sku_list)
    
    # Tampilkan info produk
    sku_info = ml_df[ml_df['SKU_ID'] == selected_sku].iloc[0]
    st.markdown(f"""
    **Produk:** {sku_info['Product Name']}  
    **Sub Brand:** {sku_info['Sub Brand']}  
    **Tier:** {sku_info['SKU Tier']}
    """)
    
    # Input untuk prediksi
    st.markdown("---")
    st.markdown("### 📊 Input untuk Prediksi")
    
    # Ambil data terakhir SKU ini
    sku_data = ml_df[ml_df['SKU_ID'] == selected_sku].sort_values('Date')
    last_row = sku_data.iloc[-1] if not sku_data.empty else None
    
    if last_row is not None:
        default_lag1 = float(last_row['Lag_1'])
        default_lag2 = float(last_row['Lag_2'])
        default_lag3 = float(last_row['Lag_3'])
        default_rolling = float(last_row['Rolling_Mean_3'])
        last_month = last_row['Month']
    else:
        default_lag1 = default_lag2 = default_lag3 = default_rolling = 100
        last_month = 1
    
    # Input features
    target_month = st.selectbox("Bulan Target", range(1, 13), index=last_month-1)
    lag_1 = st.number_input("Penjualan Bulan Lalu", min_value=0, value=int(default_lag1))
    lag_2 = st.number_input("Penjualan 2 Bulan Lalu", min_value=0, value=int(default_lag2))
    lag_3 = st.number_input("Penjualan 3 Bulan Lalu", min_value=0, value=int(default_lag3))
    rolling_mean = st.number_input("Rata-rata 3 Bulan", min_value=0, value=int(default_rolling))

# ---------- Main Content ----------
st.title("📈 Sales Forecasting AI - Inventory Edition")
st.markdown("Memprediksi penjualan bulanan berdasarkan data historis SKU")

# Tabs untuk navigasi
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "📅 Data Historis", "📊 Analisis SKU", "ℹ️ Info"])

with tab1:
    st.subheader(f"Prediksi untuk SKU: {selected_sku}")
    
    # Buat prediction input
    pred_features = pd.DataFrame([{
        'Month': target_month,
        'Lag_1': lag_1,
        'Lag_2': lag_2,
        'Lag_3': lag_3,
        'Rolling_Mean_3': rolling_mean
    }])
    
    # Prediksi
    if model == "dummy":
        prediction = dummy_predict(sku_data, target_month, lag_1, lag_2, lag_3, rolling_mean)
        st.info("Menggunakan model dummy karena model terlatih tidak ditemukan.")
    else:
        try:
            # Sesuaikan dengan nama fitur yang digunakan model Anda
            prediction = model.predict(pred_features)[0]
        except:
            prediction = dummy_predict(sku_data, target_month, lag_1, lag_2, lag_3, rolling_mean)
            st.warning("Model gagal memprediksi, menggunakan fallback dummy.")
    
    # Tampilkan prediksi
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediksi Penjualan", f"{prediction:,.0f} unit")
    
    with col2:
        if last_row is not None:
            last_actual = last_row['Sales']
            change = ((prediction - last_actual) / last_actual) * 100
            st.metric("Perubahan vs Bulan Lalu", f"{change:.1f}%", delta=f"{change:.1f}%")
    
    with col3:
        st.metric("Bulan Target", datetime(1900, target_month, 1).strftime('%B'))
    
    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Forecast", 'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [0, max(1000, prediction*1.5)], 'tickcolor': 'white'},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, prediction*0.7], 'color': "rgba(255,255,255,0.1)"},
                {'range': [prediction*0.7, prediction*1.3], 'color': "rgba(255,255,255,0.2)"}
            ]
        }
    ))
    fig_gauge.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
    st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.subheader(f"Data Historis - {selected_sku}")
    
    # Tampilkan data historis untuk SKU terpilih
    sku_history = full_df[full_df['SKU_ID'] == selected_sku].sort_values('Date')
    
    # Line chart historis
    fig_hist = px.line(
        sku_history, 
        x='Date', 
        y='Sales',
        title=f'Tren Penjualan {sku_info["Product Name"]}',
        markers=True
    )
    fig_hist.update_layout(template='plotly_dark')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Tabel data
    st.dataframe(
        sku_history[['Date', 'Month', 'Year', 'Sales', 'Lag_1', 'Lag_2', 'Lag_3']].tail(12),
        use_container_width=True
    )

with tab3:
    st.subheader("Analisis Perbandingan SKU")
    
    # Aggregasi per SKU
    sku_summary = ml_df.groupby(['SKU_ID', 'Product Name', 'Sub Brand']).agg({
        'Sales': ['mean', 'sum', 'max', 'min'],
        'Date': 'count'
    }).round(0)
    sku_summary.columns = ['Rata-rata', 'Total', 'Max', 'Min', 'Jumlah_Bulan']
    sku_summary = sku_summary.reset_index()
    
    # Bar chart perbandingan
    top_skus = sku_summary.nlargest(10, 'Total')
    fig_comp = px.bar(
        top_skus,
        x='Product Name',
        y='Total',
        color='Sub Brand',
        title='Top 10 SKU berdasarkan Total Penjualan'
    )
    fig_comp.update_layout(template='plotly_dark')
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Tabel summary
    st.dataframe(sku_summary, use_container_width=True)

with tab4:
    st.subheader("Informasi Dataset")
    st.markdown(f"""
    *   **Sumber Data**: Google Sheet  
    *   **Total SKU**: {ml_df['SKU_ID'].nunique()}  
    *   **Total Records**: {len(ml_df)}  
    *   **Periode Data**: {ml_df['Date'].min().strftime('%b %Y')} - {ml_df['Date'].max().strftime('%b %Y')}  
    *   **Fitur yang Digunakan**:
        *   Month (bulan ke-1 s/d 12)
        *   Lag_1, Lag_2, Lag_3 (penjualan 1,2,3 bulan sebelumnya)
        *   Rolling_Mean_3 (rata-rata bergerak 3 bulan)
    """)
    
    # Sample data
    st.markdown("### Contoh Data Siap ML")
    st.dataframe(ml_df.head(10), use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data dari Google Sheets")
