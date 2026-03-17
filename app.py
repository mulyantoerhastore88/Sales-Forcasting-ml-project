import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import traceback

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="SalesForecast AI - Inventory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: radial-gradient(circle at 20% 20%, #1a1a2e, #0a0a0f); color: #ffffff; }
    .glass-card {
        background: rgba(20, 20, 30, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 30px;
        padding: 30px;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    .glass-card:hover { transform: translateY(-5px); border-color: rgba(59,130,246,0.3); }
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
    }
    .metric-label { color: #8a8a9e; font-size: 1.1rem; text-transform: uppercase; }
    h1, h2, h3 { background: linear-gradient(135deg, #ffffff, #b0c4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    hr { border-color: rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
@st.cache_data(ttl=1800)
def load_data_from_gsheet():
    """Load data from Google Sheets."""
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if 'gcp_service_account' not in st.secrets:
            st.error("Key 'gcp_service_account' tidak ditemukan di secrets!")
            return None

        cred_dict = st.secrets["gcp_service_account"]
        required = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email',
                    'client_id', 'auth_uri', 'token_uri', 'auth_provider_x509_cert_url', 'client_x509_cert_url']
        if any(f not in cred_dict for f in required):
            st.error("Secrets tidak lengkap. Periksa file secrets.toml")
            return None

        # Bersihkan private key
        private_key = cred_dict["private_key"].strip().strip('"').strip("'").replace('\\n', '\n')
        if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
            private_key = '-----BEGIN PRIVATE KEY-----\n' + private_key
        if not private_key.endswith('-----END PRIVATE KEY-----'):
            private_key = private_key + '\n-----END PRIVATE KEY-----'

        credentials = {**cred_dict, "private_key": private_key}
        creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials, scope)
        client = gspread.authorize(creds)

        sheet_key = '1PuoII49N-IWOaNO8fSMYGwuvFf1T68_Kez30WN9q8Ds'
        sheet = client.open_by_key(sheet_key)

        # Coba ambil worksheet 'Sales'
        try:
            worksheet = sheet.worksheet("Sales")
        except:
            worksheets = [w.title for w in sheet.worksheets()]
            st.error(f"Worksheet 'Sales' tidak ditemukan. Yang tersedia: {worksheets}")
            return None

        data = worksheet.get_all_values()
        if len(data) < 2:
            st.error("Data kosong atau hanya header")
            return None

        headers = [h.strip() if h else f"Col_{i}" for i, h in enumerate(data[0])]
        df = pd.DataFrame(data[1:], columns=headers).dropna(how='all')
        st.success(f"✅ Berhasil memuat {len(df)} baris, {len(headers)} kolom")
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_data
def prepare_data_for_forecasting(df):
    """Transform data from wide to long and create features."""
    if df is None or df.empty:
        return None, None

    # Identifikasi kolom ID
    id_cols = [c for c in ['No', 'SKU GOA', 'SKU_ID', 'Product Name', 'Notes', 'Sub Brand', 'SKU Tier'] if c in df.columns]
    if not id_cols:
        id_cols = [df.columns[0]]

    # Identifikasi kolom bulan (MMM YY)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    years = ['24','25','26','2024','2025','2026']
    month_cols = []
    for col in df.columns:
        if col not in id_cols and any(m in col for m in months) and any(y in col for y in years):
            month_cols.append(col)
    if not month_cols:
        month_cols = [c for c in df.columns if c not in id_cols]

    # Melt
    df_long = pd.melt(df, id_vars=id_cols, value_vars=month_cols, var_name='Month_Year', value_name='Sales')
    df_long['Sales'] = pd.to_numeric(df_long['Sales'], errors='coerce').fillna(0)

    def parse_date(mmyy):
        for fmt in ['%b %y', '%b_%y', '%b-%y', '%B %y', '%b %Y', '%B %Y']:
            try:
                return datetime.strptime(str(mmyy).strip(), fmt)
            except:
                pass
        return None
    df_long['Date'] = df_long['Month_Year'].apply(parse_date)
    df_long = df_long.dropna(subset=['Date'])

    if df_long.empty:
        return None, None

    sku_col = id_cols[1] if len(id_cols) > 1 else id_cols[0]
    df_long = df_long.sort_values([sku_col, 'Date']).reset_index(drop=True)

    # Feature engineering
    df_long['Month'] = df_long['Date'].dt.month
    df_long['Year'] = df_long['Date'].dt.year
    df_long['Quarter'] = df_long['Date'].dt.quarter

    df_long['Lag_1'] = df_long.groupby(sku_col)['Sales'].shift(1)
    df_long['Lag_2'] = df_long.groupby(sku_col)['Sales'].shift(2)
    df_long['Lag_3'] = df_long.groupby(sku_col)['Sales'].shift(3)
    df_long['Rolling_Mean_3'] = df_long.groupby(sku_col)['Sales'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    df_ml = df_long.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3']).reset_index(drop=True)
    return df_ml, df_long

@st.cache_resource
def load_model():
    """Load trained model or return dummy."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(base_dir, 'sales_model.pkl'), os.path.join(base_dir, 'model.pkl')]:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                continue
    return "dummy"

def simple_predict(month, lag1, lag2, lag3, roll_mean):
    """Simple rule-based prediction."""
    weights = {'lag1':0.4, 'lag2':0.3, 'lag3':0.2, 'roll':0.1}
    pred = weights['lag1']*lag1 + weights['lag2']*lag2 + weights['lag3']*lag3 + weights['roll']*roll_mean
    seasonal = {1:1.1,2:0.9,3:1.0,4:1.0,5:1.1,6:1.0,7:0.9,8:0.9,9:1.0,10:1.1,11:1.2,12:1.3}
    pred *= seasonal.get(month, 1.0)
    return max(0, round(pred, 0))

# ---------- Main App ----------
def main():
    st.title("📈 Sales Forecasting AI - Inventory Edition")
    st.markdown("Memprediksi penjualan bulanan berdasarkan data historis SKU")

    with st.status("📡 Menghubungkan ke Google Sheets...", expanded=True) as status:
        raw_df = load_data_from_gsheet()
        if raw_df is None:
            status.update(label="❌ Gagal memuat data", state="error")
            st.stop()
        ml_df, full_df = prepare_data_for_forecasting(raw_df)
        if ml_df is None or full_df is None:
            status.update(label="❌ Gagal memproses data", state="error")
            st.stop()
        status.update(label="✅ Data siap!", state="complete", expanded=False)

    model = load_model()

    # Tentukan kolom SKU
    sku_col = next((c for c in ['SKU_ID', 'SKU ID', 'SKU GOA', 'SKU'] if c in ml_df.columns), ml_df.columns[1])

    # Sidebar
    with st.sidebar:
        st.markdown("<br><h2 style='text-align: center;'>🔍 Filter Data</h2>", unsafe_allow_html=True)
        sku_list = sorted(ml_df[sku_col].unique())
        selected_sku = st.selectbox("Pilih SKU", sku_list)

        sku_data = ml_df[ml_df[sku_col] == selected_sku].sort_values('Date')
        if not sku_data.empty:
            last = sku_data.iloc[-1]
            default_lag1 = float(last['Lag_1']) if pd.notna(last['Lag_1']) else 100
            default_lag2 = float(last['Lag_2']) if pd.notna(last['Lag_2']) else 100
            default_lag3 = float(last['Lag_3']) if pd.notna(last['Lag_3']) else 100
            default_roll = float(last['Rolling_Mean_3']) if pd.notna(last['Rolling_Mean_3']) else 100
            last_month = int(last['Month']) if pd.notna(last['Month']) else 1
            last_sales = float(last['Sales']) if pd.notna(last['Sales']) else 0

            st.markdown("---")
            st.markdown("### 📊 Input Prediksi")
            target_month = st.selectbox("Bulan Target", range(1,13), index=last_month-1,
                                        format_func=lambda x: datetime(1900,x,1).strftime('%B'))
            lag1 = st.number_input("Penjualan Bulan Lalu", min_value=0, value=int(default_lag1))
            lag2 = st.number_input("Penjualan 2 Bulan Lalu", min_value=0, value=int(default_lag2))
            lag3 = st.number_input("Penjualan 3 Bulan Lalu", min_value=0, value=int(default_lag3))
            roll = st.number_input("Rata-rata 3 Bulan", min_value=0, value=int(default_roll))
        else:
            st.warning("Tidak ada data SKU ini")
            target_month, lag1, lag2, lag3, roll, last_sales = 1, 100, 100, 100, 100, 0

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "📅 Data Historis", "📊 Analisis SKU", "ℹ️ Info Dataset"])

    with tab1:
        st.subheader(f"Prediksi untuk SKU: {selected_sku}")
        if model == "dummy":
            pred = simple_predict(target_month, lag1, lag2, lag3, roll)
            st.info("ℹ️ Menggunakan aturan sederhana (model tidak ditemukan)")
        else:
            try:
                pred = model.predict(pd.DataFrame([{'Month':target_month, 'Lag_1':lag1, 'Lag_2':lag2, 'Lag_3':lag3, 'Rolling_Mean_3':roll}]))[0]
            except:
                pred = simple_predict(target_month, lag1, lag2, lag3, roll)
                st.warning("Model gagal, gunakan fallback")

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediksi", f"{pred:,.0f} unit")
        if last_sales > 0:
            change = (pred - last_sales)/last_sales*100
            col2.metric("Perubahan", f"{change:+.1f}%", delta=f"{change:+.1f}%")
        else:
            col2.metric("Perubahan", "N/A")
        col3.metric("Bulan Target", datetime(1900,target_month,1).strftime('%B'))

        # Gauge
        max_val = max(1000, pred*2)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':"Forecast",'font':{'color':'white'}},
            gauge={'axis':{'range':[0,max_val],'tickcolor':'white'},'bar':{'color':"#3b82f6"},
                   'steps':[{'range':[0,max_val*0.3],'color':"rgba(255,255,255,0.1)"},
                            {'range':[max_val*0.3,max_val*0.7],'color':"rgba(255,255,255,0.2)"},
                            {'range':[max_val*0.7,max_val],'color':"rgba(255,255,255,0.3)"}]}))
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader(f"Data Historis - {selected_sku}")
        hist = full_df[full_df[sku_col] == selected_sku].sort_values('Date')
        if not hist.empty:
            fig = px.line(hist, x='Date', y='Sales', markers=True, title='Tren Penjualan')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(hist[['Date','Sales','Lag_1','Lag_2','Lag_3','Rolling_Mean_3']].tail(24), use_container_width=True)
        else:
            st.warning("Tidak ada data")

    with tab3:
        st.subheader("Analisis Perbandingan SKU")
        group_cols = [sku_col] + [c for c in ['Product Name','Sub Brand','SKU Tier'] if c in ml_df.columns]
        summary = ml_df.groupby(group_cols)['Sales'].agg(['mean','sum','max','min','count']).round(0).reset_index()
        summary.columns = group_cols + ['Rata-rata','Total','Max','Min','Jumlah_Bulan']
        summary = summary.sort_values('Total', ascending=False)
        st.dataframe(summary, use_container_width=True)

        # Bar chart top 15
        top = summary.head(15)
        label = 'Product Name' if 'Product Name' in top.columns else sku_col
        fig = px.bar(top, x=label, y='Total', color='Sub Brand' if 'Sub Brand' in top.columns else None,
                     title='Top 15 SKU berdasarkan Total Penjualan', text_auto='.0f')
        fig.update_layout(template='plotly_dark', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = summary.to_csv(index=False)
        st.download_button("📥 Download Ringkasan SKU", csv, "sku_summary.csv", "text/csv")

    with tab4:
        st.subheader("Informasi Dataset")
        if full_df is not None:
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("Total SKU", full_df[sku_col].nunique())
            col2.metric("Total Records", len(full_df))
            col3.metric("Periode Awal", full_df['Date'].min().strftime('%b %Y'))
            col4.metric("Periode Akhir", full_df['Date'].max().strftime('%b %Y'))

            st.markdown("**Contoh Data Siap ML**")
            st.dataframe(ml_df[['Date',sku_col,'Sales','Lag_1','Lag_2','Lag_3']].head(10), use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#666'>Built with ❤️ using Streamlit | Data dari Google Sheets | © 2026</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
