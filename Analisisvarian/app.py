import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Init Groq client
client = Groq(api_key=GROQ_API_KEY)

# Streamlit App UI
st.set_page_config(page_title="Financial Copilot AI", page_icon="📊", layout="wide")
st.title("📊 Financial Copilot AI – Scenario Planning & Strategic Insights")
st.write("Upload financial data and enter a scenario prompt to simulate different projections!")

# Model selector
selected_model = st.selectbox(
    "🤖 Select AI Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    index=0
)

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    st.subheader("📌 Financial Ratio Analysis")

required_cols = [
    "Current Assets", "Current Liabilities",
    "Total Assets", "Total Liabilities",
    "Equity", "Revenue", "COGS",
    "Net Income", "Interest Expense", "Inventory", "Receivables"
]

if all(col in df.columns for col in required_cols):

    # LIQUIDITY
    df["Current Ratio"] = df["Current Assets"] / df["Current Liabilities"]
    df["Quick Ratio"] = (df["Current Assets"] - df["Inventory"]) / df["Current Liabilities"]
    df["Cash Ratio"] = df["Current Assets"] * 0.2 / df["Current Liabilities"]  # asumsi kas 20%

    # SOLVENCY
    df["Debt to Asset"] = df["Total Liabilities"] / df["Total Assets"]
    df["Debt to Equity"] = df["Total Liabilities"] / df["Equity"]
    df["Interest Coverage"] = (df["Net Income"] + df["Interest Expense"]) / df["Interest Expense"]

    # PROFITABILITY
    df["ROA (%)"] = (df["Net Income"] / df["Total Assets"]) * 100
    df["ROE (%)"] = (df["Net Income"] / df["Equity"]) * 100
    df["Net Profit Margin (%)"] = (df["Net Income"] / df["Revenue"]) * 100
    df["Gross Profit Margin (%)"] = ((df["Revenue"] - df["COGS"]) / df["Revenue"]) * 100

    # ACTIVITY
    df["Total Asset Turnover"] = df["Revenue"] / df["Total Assets"]
    df["Inventory Turnover"] = df["COGS"] / df["Inventory"]
    df["Receivable Turnover"] = df["Revenue"] / df["Receivables"]

    # Show table
    st.dataframe(df[[
        "Current Ratio","Quick Ratio","Cash Ratio",
        "Debt to Asset","Debt to Equity","Interest Coverage",
        "ROA (%)","ROE (%)","Net Profit Margin (%)","Gross Profit Margin (%)",
        "Total Asset Turnover","Inventory Turnover","Receivable Turnover"
    ]])

    # Charts
    st.subheader("📈 Liquidity Ratios")
    st.line_chart(df[["Current Ratio","Quick Ratio","Cash Ratio"]])

    st.subheader("📈 Solvency Ratios")
    st.line_chart(df[["Debt to Asset","Debt to Equity","Interest Coverage"]])

    st.subheader("📈 Profitability Ratios")
    st.line_chart(df[["ROA (%)","ROE (%)","Net Profit Margin (%)"]])

    st.subheader("📈 Activity Ratios")
    st.line_chart(df[["Total Asset Turnover","Inventory Turnover","Receivable Turnover"]])

else:
    st.warning("Dataset belum memiliki semua kolom untuk menghitung rasio keuangan.")


    # Check for required columns
    required_columns = ["Category", "Base Forecast"]
    if not all(col in df.columns for col in required_columns):
        st.error("⚠️ The uploaded file must contain 'Category' and 'Base Forecast' columns!")
        st.stop()

    # Scenario Input
    scenario_prompt = st.text_area(
        "📝 Enter a financial scenario (e.g., 'Revenue drops 10%', 'Costs increase by 5%'):"
    )

    if st.button("🚀 Generate Scenarios"):
        # Generate Different Scenario Projections
        df["Optimistic"] = df["Base Forecast"] * np.random.uniform(1.1, 1.3, len(df))
        df["Pessimistic"] = df["Base Forecast"] * np.random.uniform(0.7, 0.9, len(df))
        df["Worst Case"] = df["Base Forecast"] * np.random.uniform(0.5, 0.7, len(df))

        # Layout: 2 columns
        col1, col2 = st.columns([2, 1])

        with col1:
            # Display scenario data
            st.subheader("📊 Scenario-Based Projections")
            st.dataframe(df)

            # Plot Scenario Analysis
            fig_scenarios = px.bar(
                df,
                x="Category",
                y=["Base Forecast", "Optimistic", "Pessimistic", "Worst Case"],
                title="📉 Scenario Planning: Financial Projections",
                barmode="group",
                text_auto=".2s",
            )
            st.plotly_chart(fig_scenarios, use_container_width=True)

        with col2:
            # AI Section
            st.subheader("🤖 AI Financial Copilot Insights")

            # AI Summary of Scenario Data (limit rows to avoid token overload)
            df_preview = df.head(20).to_string(index=False)

            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an AI Financial Copilot.
                            You help analyze scenario planning, variance analysis, and strategy.
                            You can discuss EBIT, revenue, margin, COGS, OPEX, CAPEX, cash flow,
                            liquidity, financial distress, firm value, ESG, risk, and growth strategy.
                            Always provide structured insights with both short-term and long-term recommendations."""
                        },
                        {
                            "role": "user",
                            "content": f"Here are the scenario projections:\n{df_preview}\nScenario: {scenario_prompt}\nPlease summarize the key insights and recommendations."
                        }
                    ],
                    model=selected_model,
                )
                st.markdown("**AI Initial Analysis:**")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"⚠️ AI request failed: {e}")

            # Persistent chat messages
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_query = st.text_input("💬 Ask your Financial Copilot anything (EBIT, Cash Flow, CAPEX, ESG, etc.)")

            col_chat1, col_chat2 = st.columns([4,1])
            with col_chat1:
                send_btn = st.button("Send")
            with col_chat2:
                reset_btn = st.button("🔄 Reset Chat")

            if reset_btn:
                st.session_state.chat_history = []
                st.success("Chat history cleared!")

            if send_btn and user_query:
                try:
                    chat_response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": """You are an AI Financial Copilot.
                                You can answer questions about EBIT, revenue, margin, OPEX, CAPEX, cash flow,
                                firm value, ESG, risks, financial distress, and strategy.
                                Always explain clearly and give actionable recommendations."""
                            },
                            *st.session_state.chat_history,
                            {"role": "user", "content": f"Dataset preview:\n{df_preview}\nScenario: {scenario_prompt}\n\nQuestion: {user_query}"}
                        ],
                        model=selected_model,
                    )

                    ai_answer = chat_response.choices[0].message.content

                    # Save to session state
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_answer})

                except Exception as e:
                    st.error(f"⚠️ AI chat request failed: {e}")

            # Show chat history
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"**👤 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Copilot:** {msg['content']}")
