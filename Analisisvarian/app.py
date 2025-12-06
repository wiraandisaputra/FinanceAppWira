import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# ===================== CONFIG =====================
st.set_page_config(
    page_title="AI Agent Analisis Laporan Keuangan",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 AI Agent Analisis Laporan Keuangan")
st.write("Upload laporan keuangan PDF / Excel dan sistem otomatis menganalisis rasio serta membuat dashboard.")

# ===================== PDF READER =====================
try:
    import pdfplumber
    PDF_ENGINE = "pdfplumber"
except:
    from PyPDF2 import PdfReader
    PDF_ENGINE = "PyPDF2"

def extract_text_from_pdf(file):
    text = ""
    if PDF_ENGINE == "pdfplumber":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
    else:
        reader = PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

    return text

# ===================== EXTRACT NUMBER =====================
def extract_value(text, keywords):
    pattern = rf"({keywords}).{{0,50}}?([\d.,]+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        value = match.group(2)
        value = value.replace('.', '').replace(',', '.')
        try:
            return float(value)
        except:
            return None
    return None

# ===================== FILE UPLOADER =====================
uploaded_file = st.file_uploader(
    "📤 Upload laporan keuangan (PDF / Excel)",
    type=["pdf", "xlsx"]
)

if uploaded_file:

    st.success("✅ File berhasil diupload")

    # ===================== EXCEL =====================
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Preview Data Excel")
        st.dataframe(df)

        # Auto picking common columns
        col_map = {col.lower(): col for col in df.columns}

        def find_col(keyword):
            for k in col_map:
                if keyword in k:
                    return col_map[k]
            return None

        current_assets_col = find_col("lancar")
        total_assets_col = find_col("aset")
        liabilities_col = find_col("liabil")
        equity_col = find_col("ekuitas")
        revenue_col = find_col("pendapatan") or find_col("penjualan")
        net_income_col = find_col("laba")

        if all([current_assets_col, total_assets_col, liabilities_col, equity_col, revenue_col, net_income_col]):

            current_assets = df[current_assets_col].sum()
            total_assets = df[total_assets_col].sum()
            total_liabilities = df[liabilities_col].sum()
            equity = df[equity_col].sum()
            revenue = df[revenue_col].sum()
            net_income = df[net_income_col].sum()

        else:
            st.error("❌ Kolom laporan keuangan tidak terdeteksi otomatis.")
            st.stop()

    # ===================== PDF =====================
    else:
        text = extract_text_from_pdf(uploaded_file)

        current_assets   = extract_value(text, "Aset Lancar|Aktiva Lancar")
        total_assets     = extract_value(text, "Total Aset|Total Aktiva")
        total_liabilities = extract_value(text, "Liabilitas|Total Liabilitas|Total Hutang")
        equity           = extract_value(text, "Ekuitas|Modal")
        revenue          = extract_value(text, "Pendapatan|Penjualan")
        net_income       = extract_value(text, "Laba Bersih|Laba Tahun Berjalan")

        if None in [current_assets, total_assets, total_liabilities, equity, revenue, net_income]:
            st.error("❌ Tidak semua data bisa diekstrak otomatis dari PDF")
            with st.expander("Check extracted text preview"):
                st.text(text[:3000])
            st.stop()

    # ===================== RATIO CALCULATION =====================
    current_ratio = current_assets / total_liabilities
    debt_ratio = total_liabilities / total_assets
    roe = net_income / equity
    roa = net_income / total_assets
    asset_turnover = revenue / total_assets

    # ===================== DISPLAY METRIC =====================
    st.subheader("📌 Ringkasan Nilai Keuangan")

    col1, col2, col3 = st.columns(3)

    col1.metric("Aset Lancar", f"{current_assets:,.0f}")
    col2.metric("Total Aset", f"{total_assets:,.0f}")
    col3.metric("Total Liabilitas", f"{total_liabilities:,.0f}")

    col1.metric("Ekuitas", f"{equity:,.0f}")
    col2.metric("Pendapatan", f"{revenue:,.0f}")
    col3.metric("Laba Bersih", f"{net_income:,.0f}")

    # ===================== RATIO TABLE =====================
    ratio_data = pd.DataFrame({
        "Rasio": [
            "Current Ratio (Likuiditas)",
            "Debt Ratio (Solvabilitas)",
            "ROE (Profitabilitas)",
            "ROA (Profitabilitas)",
            "Total Asset Turnover (Aktivitas)"
        ],
        "Nilai": [
            round(current_ratio,2),
            round(debt_ratio,2),
            round(roe,2),
            round(roa,2),
            round(asset_turnover,2)
        ]
    })

    st.subheader("📊 Tabel Rasio Keuangan")
    st.dataframe(ratio_data)

    # ===================== GRAPH DASHBOARD =====================
    st.subheader("📈 Dashboard Grafik Rasio")

    fig = px.bar(
        ratio_data,
        x="Rasio",
        y="Nilai",
        title="Visualisasi Rasio Keuangan",
        text_auto=True
    )

    fig.update_layout(
        xaxis_tickangle=-40,
        height = 500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===================== ANALISIS OTOMATIS =====================
    st.subheader("🧠 Interpretasi Otomatis")

    analisis = f"""
    **1. Likuiditas:**
    Current Ratio = {round(current_ratio,2)}  
    Jika >1 artinya perusahaan memiliki kemampuan cukup untuk membayar kewajiban jangka pendek.

    **2. Solvabilitas:**
    Debt Ratio = {round(debt_ratio,2)}  
    Semakin rendah semakin baik karena hutang lebih kecil dibanding aset.

    **3. Profitabilitas:**
    ROE = {round(roe,2)}  
    ROA = {round(roa,2)}  
    Menunjukkan kemampuan aset dan modal menghasilkan laba.

    **4. Aktivitas:**
    Asset Turnover = {round(asset_turnover,2)}  
    Efisiensi aset dalam menghasilkan pendapatan.
    """

    st.markdown(analisis)

    st.success("✅ Analisis selesai. Siap dipakai untuk tugas / skripsi / publikasi.")
