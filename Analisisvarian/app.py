import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="AI Financial Analysis", layout="wide")

st.title("📊 AI Agent Analisis Laporan Keuangan dari PDF")
st.write("Upload laporan keuangan (PDF) untuk dianalisis otomatis")

# -----------------------------------
# Fungsi mengambil angka dari PDF
# -----------------------------------
def extract_number(text, keyword):
    pattern = rf"{keyword}.*?([\d,\.]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    return 0

# -----------------------------------
# Upload PDF
# -----------------------------------
uploaded_file = st.file_uploader("Upload Laporan Keuangan (PDF)", type="pdf")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()

    # Ambil data penting (ubah kata sesuai laporan BEI)
    current_assets = extract_number(all_text, "Aset Lancar")
    current_liabilities = extract_number(all_text, "Liabilitas Jangka Pendek")
    total_assets = extract_number(all_text, "Total Aset")
    total_liabilities = extract_number(all_text, "Total Liabilitas")
    equity = extract_number(all_text, "Total Ekuitas")
    net_income = extract_number(all_text, "Laba Bersih")
    revenue = extract_number(all_text, "Pendapatan")

    # -----------------------------------
    # Tampilkan data hasil ekstraksi
    # -----------------------------------
    st.subheader("📌 Data Utama yang Terbaca")
    data = {
        "Aset Lancar": current_assets,
        "Liabilitas Lancar": current_liabilities,
        "Total Aset": total_assets,
        "Total Liabilitas": total_liabilities,
        "Ekuitas": equity,
        "Laba Bersih": net_income,
        "Pendapatan": revenue
    }

    st.dataframe(pd.DataFrame(data.items(), columns=["Komponen", "Jumlah (Rp)"]))

    # -----------------------------------
    # Hitung RASIO
    # -----------------------------------
    st.subheader("📊 Hasil Rasio Keuangan")

    # Likuiditas
    current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0

    # Solvabilitas
    debt_ratio = total_liabilities / total_assets if total_assets != 0 else 0
    der = total_liabilities / equity if equity != 0 else 0

    # Profitabilitas
    roa = net_income / total_assets if total_assets != 0 else 0
    roe = net_income / equity if equity != 0 else 0
    npm = net_income / revenue if revenue != 0 else 0

    # Aktivitas
    total_asset_turnover = revenue / total_assets if total_assets != 0 else 0


    ratio_data = {
        "Current Ratio": current_ratio,
        "Debt Ratio": debt_ratio,
        "DER": der,
        "ROA": roa,
        "ROE": roe,
        "NPM": npm,
        "Total Asset Turnover": total_asset_turnover
    }

    st.dataframe(pd.DataFrame(ratio_data.items(), columns=["Rasio", "Nilai"]))


    # -----------------------------------
    # Grafik
    # -----------------------------------

    st.subheader("📈 Grafik Rasio")

    fig, ax = plt.subplots()
    ax.bar(ratio_data.keys(), ratio_data.values())
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # -----------------------------------
    # AI INSIGHT (Analisis Otomatis)
    # -----------------------------------
    st.subheader("🤖 AI Insight")

    if current_ratio > 1:
        likuiditas_status = "Likuid (kondisi baik)"
    else:
        likuiditas_status = "Kurang likuid"

    if debt_ratio < 0.6:
        solvabilitas_status = "Struktur modal sehat"
    else:
        solvabilitas_status = "Risiko utang cukup tinggi"

    if roa > 0.05:
        profitabilitas_status = "Perusahaan cukup profitable"
    else:
        profitabilitas_status = "Profitabilitas rendah"


    st.write(f"""
    • Rasio Likuiditas: {likuiditas_status} (CR = {current_ratio:.2f})
    • Rasio Solvabilitas: {solvabilitas_status} (Debt Ratio = {debt_ratio:.2f})
    • Rasio Profitabilitas: {profitabilitas_status} (ROA = {roa:.2f})
    • Aktivitas Aset: Perputaran aset sebesar {total_asset_turnover:.2f} kali
    """)
