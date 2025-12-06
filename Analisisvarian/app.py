import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# =====================
# SETUP
# =====================
st.set_page_config(page_title="AI Agent Akuntansi", page_icon="📊", layout="wide")
st.title("📊 AI Agent Akuntansi – Analisis Laporan Keuangan Otomatis")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("API Key belum ada di .env atau Streamlit Secrets")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# =====================
# MODEL
# =====================
model_choice = st.selectbox(
    "🤖 Pilih AI Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
)

# =====================
# UPLOAD FILE
# =====================
uploaded_file = st.file_uploader(
    "📂 Upload Laporan Keuangan (Excel / PDF)",
    type=["xlsx", "pdf"]
)

# =====================
# FUNCTIONS
# =====================

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text


def calculate_ratios(data):
    ratios = {}

    try:
        ratios["Current Ratio"] = data["Current Assets"] / data["Current Liabilities"]
        ratios["Debt to Equity Ratio"] = data["Total Liabilities"] / data["Total Equity"]
        ratios["Net Profit Margin (%)"] = (data["Net Income"] / data["Revenue"]) * 100
        ratios["Total Asset Turnover"] = data["Revenue"] / data["Total Assets"]
    except:
        return None

    return ratios


def create_ratio_dataframe(ratios):
    df = pd.DataFrame(list(ratios.items()), columns=["Rasio", "Nilai"])
    return df


# =====================
# MAIN PROCESS
# =====================

if uploaded_file:

    file_type = uploaded_file.name.split(".")[-1]

    # ============= PDF =============
    if file_type == "pdf":
        st.info("📄 File PDF terdeteksi - mengekstrak teks...")

        pdf_text = extract_text_from_pdf(uploaded_file)

        st.subheader("📑 Preview Text dari PDF")
        st.text_area("Isi Laporan", pdf_text[:3000], height=250)

        if st.button("🔍 Analisis AI dari PDF"):
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst and accountant."},
                    {"role": "user", "content": f"Analisis laporan keuangan berikut dari PDF:\n{pdf_text[:4000]}"}
                ],
                model=model_choice
            )

            st.subheader("🤖 Hasil Analisis AI")
            st.write(response.choices[0].message.content)

    # ============= EXCEL =============
    else:
        df = pd.read_excel(uploaded_file)

        st.subheader("📊 Preview Data")
        st.dataframe(df)

        needed_cols = [
            "Current Assets",
            "Current Liabilities",
            "Total Liabilities",
            "Total Equity",
            "Revenue",
            "Net Income",
            "Total Assets"
        ]

        if not all(col in df.columns for col in needed_cols):
            st.warning(f"Kolom berikut harus ada: {needed_cols}")
        else:
            row = df.iloc[0]

            financial_data = {
                "Current Assets": row["Current Assets"],
                "Current Liabilities": row["Current Liabilities"],
                "Total Liabilities": row["Total Liabilities"],
                "Total Equity": row["Total Equity"],
                "Revenue": row["Revenue"],
                "Net Income": row["Net Income"],
                "Total Assets": row["Total Assets"],
            }

            ratios = calculate_ratios(financial_data)

            if ratios:
                st.subheader("📌 Hasil Perhitungan Rasio")

                ratio_df = create_ratio_dataframe(ratios)
                st.dataframe(ratio_df)

                fig = px.bar(
                    ratio_df,
                    x="Rasio",
                    y="Nilai",
                    title="Grafik Analisis Rasio Keuangan",
                    text_auto=".2f",
                    color="Rasio"
                )

                st.plotly_chart(fig, use_container_width=True)

                if st.button("🤖 Analisis AI + Rekomendasi"):
                    ai_prompt = f"""
                    Berikut ini hasil rasio keuangan perusahaan:
                    {ratio_df.to_string(index=False)}

                    Buatkan analisis:
                    1. Kondisi keuangan perusahaan
                    2. Risiko (likuiditas & solvabilitas)
                    3. Prospek jangka panjang
                    4. Rekomendasi strategi
                    """

                    response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a senior financial analyst."},
                            {"role": "user", "content": ai_prompt}
                        ],
                        model=model_choice
                    )

                    st.subheader("📈 Insight dari AI Agent Akuntansi")
                    st.write(response.choices[0].message.content)

