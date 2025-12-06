import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

# ---------- PDF ENGINE SAFE IMPORT ----------
try:
    import pdfplumber
    PDF_ENGINE = "pdfplumber"
except:
    from PyPDF2 import PdfReader
    PDF_ENGINE = "pypdf2"


st.set_page_config(
    page_title="AI Financial Agent Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Agent – Analisis Laporan Keuangan (PDF)")
st.write("Upload laporan keuangan (PDF) → Otomatis jadi Dashboard & Rasio")

# ==================== FUNCTIONS ======================

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


def extract_value(text, keyword):
    """
    Cari nilai finansial berdasarkan keyword di laporan
    """
    pattern = rf"{keyword}[^0-9\-]*([\d\.,]+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        value = match.group(1)
        value = value.replace('.', '').replace(',', '.')
        try:
            return float(value)
        except:
            return 0
    return 0.0


# =================== UPLOAD PDF ======================
uploaded_file = st.file_uploader(
    "📂 Upload Laporan Keuangan (PDF)", type=["pdf"]
)

if uploaded_file:

    with st.spinner("📄 Membaca file PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if text == "":
        st.error("❌ Tidak dapat membaca teks dari PDF")
        st.stop()

    st.success("✅ File berhasil dibaca")

    # Tampilkan preview
    with st.expander("🔍 Preview Teks (Awal Dokumen)"):
        st.text(text[:1500])

    # ================== DETEKSI DATA ==================

    current_assets = extract_value(text, "Aset Lancar")
    current_liabilities = extract_value(text, "Liabilitas Jangka Pendek|Liabilitas Lancar|Utang Lancar")
    total_assets = extract_value(text, "Total Aset")
    total_liabilities = extract_value(text, "Total Liabilitas")
    equity = extract_value(text, "Total Ekuitas|Ekuitas")
    revenue = extract_value(text, "Pendapatan|Penjualan Bersih|Revenue")
    net_income = extract_value(text, "Laba Bersih|Profit Setelah Pajak")

    data = {
        "Aset Lancar": current_assets,
        "Liabilitas Lancar": current_liabilities,
        "Total Aset": total_assets,
        "Total Liabilitas": total_liabilities,
        "Ekuitas": equity,
        "Pendapatan": revenue,
        "Laba Bersih": net_income
    }

    df_data = pd.DataFrame(data.items(), columns=["Komponen", "Nilai (Rp)"])

    st.subheader("📌 Data Keuangan yang Terdeteksi")
    st.dataframe(df_data, use_container_width=True)

    # ================== RASIO ==================

    # Likuiditas
    current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 0

    # Solvabilitas
    debt_ratio = total_liabilities / total_assets if total_assets > 0 else 0
    der = total_liabilities / equity if equity > 0 else 0

    # Profitabilitas
    roa = net_income / total_assets if total_assets > 0 else 0
    roe = net_income / equity if equity > 0 else 0
    npm = net_income / revenue if revenue > 0 else 0

    # Aktivitas
    tat = revenue / total_assets if total_assets > 0 else 0

    ratios = {
        "Current Ratio": round(current_ratio, 3),
        "Debt Ratio": round(debt_ratio, 3),
        "Debt to Equity (DER)": round(der, 3),
        "Return on Assets (ROA)": round(roa, 3),
        "Return on Equity (ROE)": round(roe, 3),
        "Net Profit Margin": round(npm, 3),
        "Total Asset Turnover": round(tat, 3)
    }

    df_ratio = pd.DataFrame(ratios.items(), columns=["Rasio", "Nilai"])

    st.subheader("📊 Rasio Keuangan Otomatis")
    st.dataframe(df_ratio, use_container_width=True)

    # ================== VISUALISASI ==================
    st.subheader("📈 Grafik Rasio")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(df_ratio["Rasio"], df_ratio["Nilai"])
    plt.xticks(rotation=45, ha='right')
    plt.title("Rasio Keuangan Perusahaan")
    plt.tight_layout()

    st.pyplot(fig)

    # ================== INTERPRETASI ==================
    st.subheader("🧠 Interpretasi Otomatis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Likuiditas",
            f"{current_ratio:.2f}",
            "Sehat" if current_ratio >= 1 else "Berisiko"
        )

    with col2:
        st.metric(
            "Solvabilitas",
            f"{debt_ratio:.2f}",
            "Sehat" if debt_ratio < 0.6 else "Risiko Tinggi"
        )

    with col3:
        st.metric(
            "Profitabilitas (ROA)",
            f"{roa:.2%}",
            "Baik" if roa > 0.05 else "Rendah"
        )

    with col4:
        st.metric(
            "Efektivitas Aset",
            f"{tat:.2f}",
            "Efektif" if tat > 1 else "Kurang Efektif"
        )

    # ================== KESIMPULAN ==================
    st.subheader("✍️ Kesimpulan Singkat Otomatis")

    if current_ratio > 1 and roa > 0.05 and debt_ratio < 0.6:
        st.success("""
        Kondisi keuangan perusahaan tergolong **stabil dan sehat**.
        Struktur modal relatif terkendali, kemampuan memenuhi kewajiban jangka pendek baik, 
        serta aset mampu menghasilkan laba secara efisien.
        """)
    elif roa <= 0 or current_ratio < 1:
        st.error("""
        Perusahaan berada dalam **kondisi berisiko secara finansial**. 
        Terdapat indikasi tekanan likuiditas dan rendahnya kinerja laba yang perlu menjadi perhatian serius manajemen.
        """)
    else:
        st.warning("""
        Perusahaan berada dalam **kondisi cukup stabil**, tetapi masih perlu meningkatkan efisiensi 
        penggunaan aset dan pengendalian liabilitas untuk memperkuat kinerja keuangan jangka panjang.
        """)
