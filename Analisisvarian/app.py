import streamlit as st
import pandas as pd
import plotly.express as px
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

st.title("🤖 AI Financial Super Agent – All-in-One Dashboard")
st.caption("Excel / PDF → Auto Analysis | Variance | Scenario | Dashboard | AI Chat | Financial Ratios")

#############################################
# PDF ENGINE
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

        st.subheader("📄 Preview Text (PDF)")
        st.text_area("PDF Content", text[:3000], height=200)

        prompt = f"""
        Berikut isi laporan keuangan (PDF):

        {text[:3000]}

        Lakukan analisis:
        - Kinerja keuangan
        - Risiko
        - Profitabilitas
        - Likuiditas (jika bisa diambil)
        - Rekomendasi strategis
        """

        with st.spinner("🤖 AI Menganalisis PDF..."):
            result = ai_analyze(prompt)

        st.subheader("🤖 AI Analysis (PDF)")
        st.write(result)

    ##################################################
    # EXCEL / CSV MODE
    ##################################################
    else:

        # MULTI SHEET READER
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            sheet_names = ["CSV File"]
            selected_sheet = "CSV File"
        else:
            sheets = pd.read_excel(uploaded_file, sheet_name=None)
            sheet_names = list(sheets.keys())
            selected_sheet = st.selectbox("📑 Pilih Sheet", sheet_names)
            df = sheets[selected_sheet]

        st.subheader(f"📊 Data dari Sheet: {selected_sheet}")
        st.dataframe(df.head(50))

        tabs = st.tabs([
            "📊 Variance Analysis",
            "📈 Scenario Planning",
            "📍 Dashboard & AI Chat",
            "📉 Analisis Rasio Keuangan ✅"
        ])

        ##################################################
        # TAB 1 - VARIANCE ANALYSIS
        ##################################################
        with tabs[0]:
            st.subheader("📊 Budget vs Actual Analysis")

            if {"Category", "Budget", "Actual"}.issubset(df.columns):

                df["Variance"] = df["Actual"] - df["Budget"]
                df["Variance %"] = (df["Variance"] / df["Budget"]) * 100

                st.dataframe(df)

                fig_bar = px.bar(
                    df,
                    x="Category",
                    y="Variance",
                    color="Variance",
                    title="Variance by Category",
                    text_auto=True
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                fig_line = px.line(
                    df,
                    x="Category",
                    y=["Budget", "Actual"],
                    markers=True,
                    title="Budget vs Actual"
                )
                st.plotly_chart(fig_line, use_container_width=True)

                with st.spinner("🤖 AI Menganalisis Variance..."):
                    prompt = f"""
                    Berikut data Budget vs Actual:

                    {df.to_string()}

                    Analisis:
                    - Penyimpangan terbesar
                    - Penyebab potensial
                    - Rekomendasi manajerial
                    """
                    ai = ai_analyze(prompt)

                st.subheader("🤖 AI Variance Commentary")
                st.write(ai)

            else:
                st.warning("Kolom wajib: Category, Budget, Actual")

        ##################################################
        # TAB 2 - SCENARIO PLANNING
        ##################################################
        with tabs[1]:
            st.subheader("📈 Scenario Planning")

            if {"Category", "Base Forecast"}.issubset(df.columns):

                scenario_prompt = st.text_area(
                    "Masukkan skenario (contoh: penurunan penjualan 10%, kenaikan biaya 5%)"
                )

                if st.button("🚀 Generate Scenario"):
                    df["Optimistic"] = df["Base Forecast"] * np.random.uniform(1.1, 1.3, len(df))
                    df["Pessimistic"] = df["Base Forecast"] * np.random.uniform(0.7, 0.9, len(df))
                    df["Worst Case"] = df["Base Forecast"] * np.random.uniform(0.5, 0.7, len(df))

                    st.dataframe(df)

                    fig = px.bar(
                        df,
                        x="Category",
                        y=["Base Forecast", "Optimistic", "Pessimistic", "Worst Case"],
                        title="Scenario Analysis",
                        barmode="group"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    with st.spinner("🤖 AI Analyze Scenario..."):
                        prompt = f"""
                        Scenario Planning Result:
                        {df.to_string()}

                        Scenario input user:
                        {scenario_prompt}

                        Berikan insight, risiko & strategi bisnis.
                        """
                        scenario_ai = ai_analyze(prompt)

                    st.subheader("🤖 AI Strategic Insight")
                    st.write(scenario_ai)

            else:
                st.warning("Kolom wajib: Category, Base Forecast")

        ##################################################
        # TAB 3 - DASHBOARD + AI CHAT
        ##################################################
        with tabs[2]:
            st.subheader("📍 Sales Dashboard")

            if {"Region", "Sales"}.issubset(df.columns):

                query = """
                SELECT Region, SUM(Sales) as Total_Sales
                FROM df
                GROUP BY Region
                ORDER BY Total_Sales DESC
                """
                region_sales = duckdb.sql(query).df()

                fig = px.bar(region_sales, x="Region", y="Total_Sales", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

                top = region_sales.iloc[0]
                low = region_sales.iloc[-1]

                st.markdown(f"""
                **Auto Insight:**
                - Tertinggi: **{top.Region}** = {top.Total_Sales:,.0f}
                - Terendah: **{low.Region}** = {low.Total_Sales:,.0f}
                """)

                with st.spinner("🤖 AI Insight..."):
                    commentary = ai_analyze(f"""
                    Berikut data penjualan per region:
                    {region_sales.to_string()}

                    Berikan analisis kinerja & strategi peningkatan.
                    """)

                st.subheader("🤖 AI Commentary")
                st.write(commentary)

                st.subheader("💬 AI Chat Mode")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = [
                        {"role": "system", "content": "Kamu adalah AI analis keuangan profesional."}
                    ]

                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    elif msg["role"] == "assistant":
                        st.chat_message("assistant").write(msg["content"])

                if question := st.chat_input("Tanyakan sesuatu tentang data..."):
                    st.session_state.chat_history.append({"role": "user", "content": question})

                    with st.spinner("🤖 Thinking..."):
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=st.session_state.chat_history
                        )

                    answer = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.chat_message("assistant").write(answer)

            else:
                st.warning("Kolom wajib: Region dan Sales")

        ##################################################
        # TAB 4 - ANALISIS RASIO KEUANGAN
        ##################################################
        with tabs[3]:
            st.subheader("📉 Analisis Rasio Keuangan Otomatis")

            try:
                data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

                def clean_number(value):
                    if value is None:
                        return 0
                    return float(str(value).replace(".", "").replace(",", ""))

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

                # PROFITABILITY
                st.markdown("## 📌 Profitability Ratios")
                gross_margin_ratio = (gross_profit / net_sales) * 100
                ebit_margin = (ebit / net_sales) * 100
                roa = (ebit / total_assets) * 100
                roe = (ebit / total_equity) * 100

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Gross Margin (%)", f"{gross_margin_ratio:.2f}%")
                col2.metric("EBIT Margin (%)", f"{ebit_margin:.2f}%")
                col3.metric("ROA (%)", f"{roa:.2f}%")
                col4.metric("ROE (%)", f"{roe:.2f}%")

                # LIQUIDITY
                st.markdown("## 📌 Liquidity Ratios")
                current_ratio = current_assets / current_liabilities
                quick_ratio = (current_assets - inventory) / current_liabilities
                cash_ratio = cash / current_liabilities

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Ratio", f"{current_ratio:.2f}")
                col2.metric("Quick Ratio", f"{quick_ratio:.2f}")
                col3.metric("Cash Ratio", f"{cash_ratio:.2f}")

                # SOLVENCY
                st.markdown("## 📌 Solvency Ratios")
                debt_to_asset = total_liabilities / total_assets
                debt_to_equity = total_liabilities / total_equity
                equity_ratio = total_equity / total_assets

                col1, col2, col3 = st.columns(3)
                col1.metric("Debt to Asset", f"{debt_to_asset:.2f}")
                col2.metric("Debt to Equity", f"{debt_to_equity:.2f}")
                col3.metric("Equity Ratio", f"{equity_ratio:.2f}")

                ratio_df = pd.DataFrame({
                    "Ratio": [
                        "Gross Margin", "EBIT Margin", "ROA", "ROE",
                        "Current Ratio", "Quick Ratio", "Cash Ratio",
                        "Debt to Asset", "Debt to Equity", "Equity Ratio"
                    ],
                    "Value": [
                        gross_margin_ratio, ebit_margin, roa, roe,
                        current_ratio, quick_ratio, cash_ratio,
                        debt_to_asset, debt_to_equity, equity_ratio
                    ]
                })

                fig_all = px.bar(
                    ratio_df,
                    x="Ratio",
                    y="Value",
                    title="📊 Financial Ratios Overview",
                    text_auto=True
                )

                st.plotly_chart(fig_all, use_container_width=True)

                with st.spinner("🤖 AI Menganalisis Semua Rasio..."):
                    prompt = f"""
                    Rasio Keuangan Perusahaan:
                    {ratio_df.to_string(index=False)}

                    Buatkan:
                    1. Analisis kondisi keuangan
                    2. Risiko utama yang terlihat
                    3. Rekomendasi strategis
                    """
                    ai_ratio = ai_analyze(prompt)

                st.subheader("🤖 AI Summary")
                st.write(ai_ratio)

            except Exception as e:
                st.error(f"❌ Error: {e}")

else:
    st.info("📂 Upload file untuk memulai analisis")
