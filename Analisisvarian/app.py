import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import duckdb
import os
from dotenv import load_dotenv

#############################################
# LOAD ENV (GROQ)
#############################################
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None

#############################################
# PAGE CONFIG
#############################################
st.set_page_config(
    page_title="AI Financial Super Agent",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 AI Financial Super Agent")
st.caption("Upload Excel / PDF → Analisis Otomatis | Rasio | Distress | AI Insight")

#############################################
# PDF READER
#############################################
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

#############################################
# AI FUNCTION
#############################################
def ai_analyze(prompt):
    if not client:
        return "⚠️ GROQ API belum diatur"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error AI: {e}"

#############################################
# CLEAN NUMBER
#############################################
def clean_number(value):
    if value is None:
        return 0
    return float(str(value).replace("Rp", "").replace(".", "").replace(",", "").strip())

#############################################
# FILE UPLOADER
#############################################
uploaded_file = st.file_uploader(
    "📂 Upload File (Excel / CSV / PDF)",
    type=["xlsx", "xls", "csv", "pdf"]
)

if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1]

    ##################################################
    # PDF MODE
    ##################################################
    if file_type == "pdf":

        text = extract_text_from_pdf(uploaded_file)

        st.subheader("📄 Isi Dokumen (PDF)")
        st.text_area("Hasil ekstraksi", text[:3000], height=250)

        with st.spinner("🤖 AI sedang menganalisis laporan PDF..."):
            prompt = f"""
            Berikut isi laporan keuangan dari file PDF:

            {text[:2000]}

            Tolong lakukan:
            1. Analisis kinerja keuangan
            2. Risiko utama perusahaan
            3. Analisis profitabilitas & likuiditas
            4. Rekomendasi strategis manajemen
            """

            result = ai_analyze(prompt)

        st.subheader("🤖 Hasil Analisis AI")
        st.write(result)

    ##################################################
    # EXCEL / CSV MODE
    ##################################################
    else:

        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            selected_sheet = "CSV"
        else:
            sheets = pd.read_excel(uploaded_file, sheet_name=None)
            selected_sheet = st.selectbox("📑 Pilih Sheet", list(sheets.keys()))
            df = sheets[selected_sheet]

        st.subheader(f"📊 Data: {selected_sheet}")
        st.dataframe(df.head(50))

        tabs = st.tabs([
            "📊 Variance Analysis",
            "📈 Scenario Planning",
            "📍 Dashboard + AI",
            "📉 Analisis Rasio Keuangan",
            "🚨 Financial Distress"
        ])

        # KONVERSI KE DICTIONARY
        data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        net_sales = clean_number(data.get("Net Sales", 0))
        gross_profit = clean_number(data.get("Gross Profit", 0))
        ebit = clean_number(data.get("EBIT", 0))
        current_assets = clean_number(data.get("Current Assets", 0))
        inventory = clean_number(data.get("Inventory", 0))
        cash = clean_number(data.get("Cash", 0))
        current_liabilities = clean_number(data.get("Current Liabilities", 0))
        total_liabilities = clean_number(data.get("Total Liabilities", 0))
        total_equity = clean_number(data.get("Total Equity", 0))
        total_assets = clean_number(data.get("Total Assets", 0))
        retained_earnings = clean_number(data.get("Retained Earnings", 0))
        market_value_equity = clean_number(data.get("Market Value Equity", 0))

        ##################################################
        # TAB 1 - VARIANCE
        ##################################################
        with tabs[0]:

            st.subheader("📊 Analisis Variance (Budget vs Actual)")

            if {"Category", "Budget", "Actual"}.issubset(df.columns):

                df["Variance"] = df["Actual"] - df["Budget"]
                df["Variance %"] = (df["Variance"] / df["Budget"]) * 100

                st.dataframe(df)

                fig = px.bar(df, x="Category", y="Variance", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

                with st.spinner("🤖 AI menganalisis penyimpangan..."):
                    st.write(ai_analyze(f"""
                    Data variance:
                    {df.to_string()}

                    Jelaskan:
                    - Penyimpangan terbesar
                    - Penyebab kemungkinan
                    - Solusi untuk manajemen
                    """))

            else:
                st.warning("Kolom wajib: Category, Budget, Actual")

        ##################################################
        # TAB 2 - SCENARIO
        ##################################################
        with tabs[1]:

            st.subheader("📈 Simulasi Skenario")

            if {"Category", "Base Forecast"}.issubset(df.columns):

                skenario = st.text_area("Masukkan skenario bisnis (contoh: penjualan turun 10%)")

                if st.button("📊 Jalankan Skenario"):

                    df["Optimis"] = df["Base Forecast"] * np.random.uniform(1.1, 1.3, len(df))
                    df["Pesimis"] = df["Base Forecast"] * np.random.uniform(0.7, 0.9, len(df))

                    st.dataframe(df)

                    fig = px.bar(df, x="Category", y=["Base Forecast", "Optimis", "Pesimis"], barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

                    with st.spinner("🤖 AI menganalisis skenario..."):
                        st.write(ai_analyze(f"""
                        Hasil simulasi:
                        {df.to_string()}

                        Skenario: {skenario}

                        Berikan: risiko & strategi
                        """))

            else:
                st.warning("Kolom wajib: Category dan Base Forecast")

        ##################################################
        # TAB 3 - DASHBOARD
        ##################################################
        with tabs[2]:

            st.subheader("📍 Dashboard Penjualan")

            if {"Region", "Sales"}.issubset(df.columns):

                region_sales = duckdb.sql("""
                    SELECT Region, SUM(Sales) as Total_Sales
                    FROM df
                    GROUP BY Region
                    ORDER BY Total_Sales DESC
                """).df()

                st.dataframe(region_sales)

                fig = px.bar(region_sales, x="Region", y="Total_Sales", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("💬 Chat dengan AI")

            if client:
                question = st.chat_input("Tanya AI tentang data keuangan...")
                if question:
                    st.write(ai_analyze(question))

        ##################################################
        # TAB 4 - RASIO KEUANGAN
        ##################################################
        with tabs[3]:

            st.subheader("📉 Analisis Rasio Keuangan")

            gross_margin = (gross_profit / net_sales) * 100 if net_sales else 0
            ebit_margin = (ebit / net_sales) * 100 if net_sales else 0
            roa = (ebit / total_assets) * 100 if total_assets else 0
            roe = (ebit / total_equity) * 100 if total_equity else 0

            current_ratio = current_assets / current_liabilities if current_liabilities else 0
            quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities else 0
            cash_ratio = cash / current_liabilities if current_liabilities else 0

            ratio_df = pd.DataFrame({
                "Rasio": [
                    "Gross Margin (%)",
                    "EBIT Margin (%)",
                    "ROA (%)",
                    "ROE (%)",
                    "Current Ratio",
                    "Quick Ratio",
                    "Cash Ratio"
                ],
                "Nilai": [
                    gross_margin,
                    ebit_margin,
                    roa,
                    roe,
                    current_ratio,
                    quick_ratio,
                    cash_ratio
                ]
            })

            st.dataframe(ratio_df)

            fig = px.bar(ratio_df, x="Rasio", y="Nilai", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

            with st.spinner("🤖 Analisis AI terhadap rasio..."):
                st.write(ai_analyze(f"""
                Data rasio keuangan:
                {ratio_df.to_string(index=False)}

                Tolong jelaskan kondisi perusahaan dan berikan rekomendasi.
                """))

        ##################################################
        # TAB 5 - DISTRESS
        ##################################################
        with tabs[4]:

            st.subheader("🚨 Peringatan Financial Distress (Altman Z-Score)")

            X1 = (current_assets - current_liabilities) / total_assets if total_assets else 0
            X2 = retained_earnings / total_assets if total_assets else 0
            X3 = ebit / total_assets if total_assets else 0
            X4 = (market_value_equity if market_value_equity else total_equity) / total_liabilities if total_liabilities else 0
            X5 = net_sales / total_assets if total_assets else 0

            if market_value_equity:
                z_score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
                safe, gray = 2.99, 1.81
            else:
                z_score = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
                safe, gray = 2.9, 1.23

            if z_score > safe:
                status = "🟢 HIJAU - AMAN"
            elif z_score >= gray:
                status = "🟡 KUNING - WASPADA"
            else:
                status = "🔴 MERAH - BAHAYA"

            st.metric("Altman Z-Score", f"{z_score:.2f}")
            st.subheader(status)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=z_score,
                title={'text': 'Risiko Kebangkrutan'},
                gauge={'axis': {'range': [-2, 6]}}
            ))

            st.plotly_chart(fig, use_container_width=True)

            with st.spinner("🤖 AI menganalisis risiko..."):
                st.write(ai_analyze(f"""
                Nilai Altman Z-Score = {z_score}
                Status = {status}

                Jelaskan risiko kebangkrutan dan langkah pencegahan.
                """))

else:
    st.info("Silakan upload laporan keuangan untuk memulai analisis")
