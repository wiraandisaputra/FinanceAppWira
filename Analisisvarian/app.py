import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import io

st.set_page_config(page_title="AI Agent Analisis Keuangan", layout="wide")
st.title("📊 AI Agent – Analisis Laporan Keuangan Otomatis")
st.write("Upload laporan keuangan dalam format **PDF atau XLSX** atau isi manual untuk mendapatkan analisis rasio & grafik.")

# ======================== FUNGSI AMBIL ANGKA =========================
def clean_number(value):
    try:
        if isinstance(value, str):
            value = value.replace(".", "").replace(",", ".")
        return float(value)
    except:
        return 0

def extract_value(text, keywords):
    if text is None:
        return 0
    lines = text.split("\n")
    for line in lines:
        for key in keywords.split("|"):
            if key.lower() in line.lower():
                numbers = re.findall(r'\d[\d.,]*', line)
                if numbers:
                    return clean_number(numbers[-1])
    return 0

# ======================== UPLOAD FILE =========================
st.subheader("📂 Upload Laporan Keuangan (PDF / XLSX)")

uploaded_file = st.file_uploader("Upload file laporan keuangan", type=["pdf","xlsx"])

pdf_text = ""
excel_data = None

if uploaded_file is not None:

    # JIKA PDF
    if uploaded_file.name.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"

            st.success("✅ PDF berhasil dibaca")

            with st.expander("Lihat sebagian isi PDF"):
                st.text(pdf_text[:4000])

        except Exception as e:
            st.error(f"Gagal membaca PDF : {e}")

    # JIKA EXCEL
    if uploaded_file.name.endswith(".xlsx"):
        try:
            excel_data = pd.read_excel(uploaded_file)
            st.success("✅ File Excel berhasil dibaca")

            with st.expander("Preview Data Excel"):
                st.dataframe(excel_data.head(20))

        except Exception as e:
            st.error(f"Gagal membaca Excel : {e}")


# ======================== INPUT MANUAL =========================
st.subheader("✏️ Input Data Manual (Jika File tidak lengkap)")

col1, col2, col3 = st.columns(3)

with col1:
    current_assets = st.number_input("Aset Lancar", min_value=0.0)
    inventory = st.number_input("Persediaan", min_value=0.0)

with col2:
    current_liabilities = st.number_input("Liabilitas Lancar", min_value=0.0)
    total_liabilities = st.number_input("Total Liabilitas", min_value=0.0)

with col3:
    total_assets = st.number_input("Total Aset", min_value=0.0)
    total_equity = st.number_input("Ekuitas", min_value=0.0)


revenue = st.number_input("Pendapatan / Penjualan", min_value=0.0)
net_income = st.number_input("Laba Bersih", min_value=0.0)


# ======================== AMBIL DARI PDF =========================
if pdf_text != "":

    if current_assets == 0:
        current_assets = extract_value(pdf_text, "Aset Lancar|Current Assets")

    if current_liabilities == 0:
        current_liabilities = extract_value(pdf_text, "Liabilitas Lancar|Utang Lancar|Current Liabilities")

    if total_assets == 0:
        total_assets = extract_value(pdf_text, "Total Aset|Total Assets")

    if total_liabilities == 0:
        total_liabilities = extract_value(pdf_text, "Total Liabilitas|Total Liabilities")

    if net_income == 0:
        net_income = extract_value(pdf_text, "Laba Bersih|Net Income|Profit")

    if revenue == 0:
        revenue = extract_value(pdf_text, "Pendapatan|Penjualan|Revenue|Sales")


# ======================== AMBIL DARI EXCEL =========================
if excel_data is not None:

    for col in excel_data.columns:
        col_str = str(col).lower()

        if "aset lancar" in col_str:
            current_assets = clean_number(excel_data[col].sum())

        if "liabilitas lancar" in col_str or "utang lancar" in col_str:
            current_liabilities = clean_number(excel_data[col].sum())

        if "total aset" in col_str:
            total_assets = clean_number(excel_data[col].sum())

        if "total liabilitas" in col_str:
            total_liabilities = clean_number(excel_data[col].sum())

        if "laba bersih" in col_str:
            net_income = clean_number(excel_data[col].sum())

        if "pendapatan" in col_str or "penjualan" in col_str:
            revenue = clean_number(excel_data[col].sum())


# ======================== ANALISIS =========================
if st.button("🚀 Analisis Rasio Keuangan"):

    if current_liabilities == 0 or total_assets == 0 or revenue == 0:
        st.error("⚠️ Data belum lengkap. Minimal isi: Aset, Liabilitas & Pendapatan")
    else:

        st.subheader("📌 HASIL ANALISIS RASIO")

        # 1. LIKUIDITAS
        current_ratio = current_assets / current_liabilities
        quick_ratio = (current_assets - inventory) / current_liabilities

        # 2. SOLVABILITAS
        debt_ratio = total_liabilities / total_assets
        debt_to_equity = total_liabilities / total_equity if total_equity != 0 else 0

        # 3. PROFITABILITAS
        net_profit_margin = (net_income / revenue) * 100
        roa = (net_income / total_assets) * 100

        # 4. AKTIVITAS
        asset_turnover = revenue / total_assets


        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Current Ratio", round(current_ratio,2))
        col1.metric("Quick Ratio", round(quick_ratio,2))

        col2.metric("Debt Ratio", round(debt_ratio,2))
        col2.metric("Debt to Equity", round(debt_to_equity,2))

        col3.metric("Net Profit Margin (%)", round(net_profit_margin,2))
        col3.metric("ROA (%)", round(roa,2))

        col4.metric("Asset Turnover", round(asset_turnover,2))


        # ======================== VISUALISASI =========================
        st.subheader("📊 Dashboard Grafik Rasio")

        data = {
            "Likuiditas": current_ratio,
            "Solvabilitas": debt_ratio,
            "Profitabilitas": net_profit_margin,
            "Aktivitas": asset_turnover
        }

        df = pd.DataFrame(list(data.items()), columns=["Rasio","Nilai"])

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(df["Rasio"], df["Nilai"])
        ax.set_title("Grafik Rasio Keuangan")
        ax.set_ylabel("Nilai Rasio")
        st.pyplot(fig)


        # ======================== KESIMPULAN =========================
        st.subheader("📢 Insight Otomatis")

        if current_ratio >= 1.5:
            st.success("Likuiditas perusahaan sangat baik")
        else:
            st.warning("Likuiditas perusahaan rendah")

        if debt_ratio <= 0.6:
            st.success("Struktur modal cukup sehat")
        else:
            st.warning("Perusahaan terlalu bergantung pada utang")

        if net_profit_margin >= 10:
            st.success("Profitabilitas sangat baik")
        else:
            st.warning("Keuntungan perusahaan rendah")

        if asset_turnover >= 1:
            st.success("Aset digunakan secara efisien")
        else:
            st.warning("Perputaran aset belum optimal")
