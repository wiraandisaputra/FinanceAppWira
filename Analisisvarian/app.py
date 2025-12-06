# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv

# Try import Groq client, but handle if not installed
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# ---------- Helper functions ----------
def load_api_key():
    # load from .env then fallback to Streamlit secrets
    load_dotenv()
    key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY") if "secrets" in dir(st) else None
    return key

def find_col(df, candidates):
    """
    Try to find a column in df given a list of candidate names.
    Matching is case-insensitive and ignores spaces/underscores.
    Returns column name if found, otherwise None.
    """
    norm = lambda s: ''.join(str(s).lower().replace(' ', '').replace('_', ''))
    df_cols_norm = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in df_cols_norm:
            return df_cols_norm[key]
    return None

def safe_div(a, b):
    # elementwise safe division, returns np.nan when denom==0 or NaN
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where((b == 0) | np.isnan(b), np.nan, a / b)
    return res

def try_create_groq_client(api_key):
    if not GROQ_AVAILABLE:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

# ---------- App UI ----------
st.set_page_config(page_title="Financial Copilot AI - Ratio Analyzer", layout="wide", page_icon="📊")
st.title("📊 Financial Copilot AI — Ratio Analyzer (Liquidity, Solvency, Profitability, Activity)")
st.write("Unggah file Excel yang berisi data keuangan per tahun. Format minimal: baris = tahun, kolom berisi nilai akun (lihat panduan di bawah).")

st.markdown("""
**Panduan singkat format data (disarankan):**  
Satu file Excel dengan satu sheet yang memiliki baris per tahun dan kolom nama akun, mis.:
`Year`, `Current Assets`, `Current Liabilities`, `Total Assets`, `Total Liabilities`, `Equity`, `Revenue`, `COGS`, `Net Income`, `Interest Expense`, `Inventory`, `Receivables`  
(Kode akan mencoba menemukan kolom yang serupa secara case-insensitive.)
""")

# Model / API setup
GROQ_API_KEY = load_api_key()
if not GROQ_API_KEY:
    st.warning("🔒 GROQ API key tidak ditemukan. Fitur AI akan non-aktif. Taruh GROQ_API_KEY di file .env atau Streamlit secrets untuk mengaktifkan AI.")
client = try_create_groq_client(GROQ_API_KEY) if GROQ_API_KEY else None

# Model selector (only show if client available otherwise disabled)
models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
if client:
    selected_model = st.selectbox("Pilih model AI (opsional)", models, index=0)
else:
    selected_model = st.selectbox("Pilih model AI (nonaktif - client Groq tidak tersedia)", models, index=0, disabled=True)

uploaded_file = st.file_uploader("📂 Upload dataset Excel (.xlsx)", type=["xlsx", "xls"])
if not uploaded_file:
    st.info("Unggah file Excel untuk memulai analisis rasio.")
    st.stop()

# Read file safely
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Gagal membaca file: {e}")
    st.stop()

# Show uploaded preview
st.subheader("Preview data (5 baris pertama)")
st.dataframe(df.head())

# Define candidate column names for each required item
candidates = {
    "Year": ["Year", "Tahun", "year"],
    "Current Assets": ["Current Assets", "Aset Lancar", "CurrentAssets", "Current_Assets"],
    "Current Liabilities": ["Current Liabilities", "Liabilitas Lancar", "CurrentLiabilities"],
    "Total Assets": ["Total Assets", "Total Aset", "TotalAssets"],
    "Total Liabilities": ["Total Liabilities", "Total Utang", "TotalLiabilities"],
    "Equity": ["Equity", "Modal", "Ekuitas"],
    "Revenue": ["Revenue", "Sales", "Penjualan"],
    "COGS": ["COGS", "HPP", "Cost of Goods Sold", "Harga Pokok Penjualan"],
    "Net Income": ["Net Income", "Laba Bersih", "NetIncome"],
    "Interest Expense": ["Interest Expense", "Beban Bunga", "Interest"],
    "Inventory": ["Inventory", "Persediaan"],
    "Receivables": ["Receivables", "Piutang"]
}

# Map found columns
found = {}
for key, cand in candidates.items():
    col = find_col(df, cand)
    found[key] = col  # may be None

missing = [k for k, v in found.items() if k != "Interest Expense" and v is None]  # interest expense optional
# We allow Interest Expense to be missing, but others required for full analysis
if missing:
    st.warning(f"⚠️ Beberapa kolom penting tidak ditemukan otomatis: {missing}. Aplikasi membutuhkan kolom-kolom tersebut untuk menghitung semua rasio.")
    st.write("Coba rename kolom Anda atau pastikan file memiliki nama kolom seperti panduan. Anda masih bisa menghitung sebagian rasio bila ada data yang cukup.")
    # show mapping to help user
st.write("Deteksi kolom (otomatis):")
st.json(found)

# Create a working DataFrame with numeric conversion
work_df = df.copy()
# If there's a Year column, try to set as index for plotting order
if found["Year"]:
    try:
        work_df["Year"] = pd.to_numeric(work_df[found["Year"]], errors="coerce").astype(pd.Int64Dtype())
    except Exception:
        work_df["Year"] = df[found["Year"]]
else:
    # if no year, create index-based year
    work_df["Year"] = np.arange(1, len(work_df) + 1)

# Convert all candidate numeric columns to float where present
for k, col in found.items():
    if col:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

# Compute ratios row-wise, guarding division by zero
# Liquidity
if found["Current Assets"] and found["Current Liabilities"]:
    work_df["Current Ratio"] = safe_div(work_df[found["Current Assets"]], work_df[found["Current Liabilities"]])
else:
    work_df["Current Ratio"] = np.nan

if found["Current Assets"] and found["Inventory"] and found["Current Liabilities"]:
    work_df["Quick Ratio"] = safe_div((work_df[found["Current Assets"]] - work_df[found["Inventory"]]), work_df[found["Current Liabilities"]])
else:
    work_df["Quick Ratio"] = np.nan

# Cash ratio: if there's no explicit Cash column, estimate as 20% of Current Assets (clear note)
if found["Current Assets"] and found["Current Liabilities"]:
    work_df["Cash Ratio"] = safe_div(work_df[found["Current Assets"]] * 0.2, work_df[found["Current Liabilities"]])
else:
    work_df["Cash Ratio"] = np.nan

# Solvency
if found["Total Liabilities"] and found["Total Assets"]:
    work_df["Debt to Asset"] = safe_div(work_df[found["Total Liabilities"]], work_df[found["Total Assets"]])
else:
    work_df["Debt to Asset"] = np.nan

if found["Total Liabilities"] and found["Equity"]:
    work_df["Debt to Equity"] = safe_div(work_df[found["Total Liabilities"]], work_df[found["Equity"]])
else:
    work_df["Debt to Equity"] = np.nan

if found["Net Income"] and found["Interest Expense"]:
    # Use (EBIT) approx: Net Income + Interest Expense (simple approx)
    work_df["Interest Coverage"] = safe_div((work_df[found["Net Income"]].fillna(0) + work_df[found["Interest Expense"]].fillna(0)), work_df[found["Interest Expense"]])
else:
    work_df["Interest Coverage"] = np.nan

# Profitability
if found["Net Income"] and found["Total Assets"]:
    work_df["ROA (%)"] = safe_div(work_df[found["Net Income"]], work_df[found["Total Assets"]]) * 100
else:
    work_df["ROA (%)"] = np.nan

if found["Net Income"] and found["Equity"]:
    work_df["ROE (%)"] = safe_div(work_df[found["Net Income"]], work_df[found["Equity"]]) * 100
else:
    work_df["ROE (%)"] = np.nan

if found["Net Income"] and found["Revenue"]:
    work_df["Net Profit Margin (%)"] = safe_div(work_df[found["Net Income"]], work_df[found["Revenue"]]) * 100
else:
    work_df["Net Profit Margin (%)"] = np.nan

if found["Revenue"] and found["COGS"]:
    work_df["Gross Profit Margin (%)"] = safe_div((work_df[found["Revenue"]] - work_df[found["COGS"]]), work_df[found["Revenue"]]) * 100
else:
    work_df["Gross Profit Margin (%)"] = np.nan

# Activity
if found["Revenue"] and found["Total Assets"]:
    work_df["Total Asset Turnover"] = safe_div(work_df[found["Revenue"]], work_df[found["Total Assets"]])
else:
    work_df["Total Asset Turnover"] = np.nan

if found["COGS"] and found["Inventory"]:
    work_df["Inventory Turnover"] = safe_div(work_df[found["COGS"]], work_df[found["Inventory"]])
else:
    work_df["Inventory Turnover"] = np.nan

if found["Revenue"] and found["Receivables"]:
    work_df["Receivable Turnover"] = safe_div(work_df[found["Revenue"]], work_df[found["Receivables"]])
else:
    work_df["Receivable Turnover"] = np.nan

# Prepare display of computed ratios
ratio_cols = [
    "Current Ratio", "Quick Ratio", "Cash Ratio",
    "Debt to Asset", "Debt to Equity", "Interest Coverage",
    "ROA (%)", "ROE (%)", "Net Profit Margin (%)", "Gross Profit Margin (%)",
    "Total Asset Turnover", "Inventory Turnover", "Receivable Turnover"
]
st.subheader("Hasil Perhitungan Rasio")
st.dataframe(work_df[["Year"] + ratio_cols].round(3))

# Charts: create an x axis from Year
x = "Year"
# Ensure Year is usable for plotting
plot_df = work_df.copy()

# Define helper to draw line charts using plotly
def plot_line(df_plot, y_cols, title):
    dfm = df_plot[[x] + y_cols].melt(id_vars=x, value_vars=y_cols, var_name="Ratio", value_name="Value")
    fig = px.line(dfm, x=x, y="Value", color="Ratio", markers=True, title=title)
    fig.update_layout(legend_title_text="Ratio", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# Layout charts in two columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Liquidity Ratios")
    plot_line(plot_df, ["Current Ratio", "Quick Ratio", "Cash Ratio"], "Liquidity Ratios Over Time")

    st.subheader("📈 Profitability Ratios")
    plot_line(plot_df, ["ROA (%)", "ROE (%)", "Net Profit Margin (%)"], "Profitability Ratios Over Time")

with col2:
    st.subheader("📈 Solvency Ratios")
    plot_line(plot_df, ["Debt to Asset", "Debt to Equity", "Interest Coverage"], "Solvency Ratios Over Time")

    st.subheader("📈 Activity Ratios")
    plot_line(plot_df, ["Total Asset Turnover", "Inventory Turnover", "Receivable Turnover"], "Activity Ratios Over Time")

# AI Insights section (optional, only if client available)
st.subheader("🤖 AI Financial Copilot (Insight & Recommendation)")

scenario_prompt = st.text_area("Masukkan ringkasan skenario / konteks (opsional):", placeholder="Contoh: Penjualan turun 10% pada 2024, biaya naik 5% ...")

if client:
    # Prepare preview of ratios to send to AI (limit rows)
    preview = plot_df[["Year"] + ratio_cols].head(10).round(3).to_string(index=False)
    system_msg = {
        "role": "system",
        "content": (
            "You are an AI Financial Copilot. You analyze ratio trends and give clear short-term and long-term recommendations."
            "Focus on liquidity, solvency, profitability and activity. Give prioritized, actionable recommendations and highlight risks."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"Here are ratio trends (year and selected ratios):\n{preview}\n\nScenario context: {scenario_prompt}\n\nPlease provide a short structured analysis (3-6 bullet points) and 2-3 recommendations."
    }

    if st.button("🔎 Generate AI Insight"):
        with st.spinner("Menghubungkan ke AI..."):
            try:
                resp = client.chat.completions.create(
                    messages=[system_msg, user_msg],
                    model=selected_model
                )
                # Groq response structure may vary; use safe access
                ai_text = ""
                try:
                    ai_text = resp.choices[0].message.content
                except Exception:
                    # fallback for different response structures
                    ai_text = str(resp)
                st.markdown("**AI Insight:**")
                st.write(ai_text)
            except Exception as e:
                st.error(f"⚠️ Permintaan AI gagal: {e}")
else:
    st.info("Fitur AI dinonaktifkan karena Groq client tidak tersedia atau API key belum diset.")

# Chat-like persistent session (simple)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
st.subheader("🗨️ Diskusi dengan Copilot (opsional)")
user_q = st.text_input("Tanyakan sesuatu terkait laporan / rasio (opsional):", "")

colq1, colq2 = st.columns([4,1])
with colq1:
    send_btn = st.button("Kirim")
with colq2:
    clear_btn = st.button("Bersihkan Riwayat")

if clear_btn:
    st.session_state.chat_history = []
    st.success("Riwayat chat dibersihkan.")

if send_btn and user_q:
    # Build messages with limited context
    preview_small = plot_df[["Year"] + ratio_cols].head(6).round(3).to_string(index=False)
    messages = [
        {"role": "system", "content": "You are a helpful Financial Copilot. Provide clear, actionable answers."},
    ]
    # append recent chat history for context (limit last 6)
    messages += st.session_state.chat_history[-6:]
    messages.append({"role": "user", "content": f"Dataset preview:\n{preview_small}\nQuestion: {user_q}"})
    if client:
        try:
            chat_resp = client.chat.completions.create(messages=messages, model=selected_model)
            try:
                ans = chat_resp.choices[0].message.content
            except Exception:
                ans = str(chat_resp)
            # save and show
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"⚠️ Permintaan chat AI gagal: {e}")
    else:
        # fallback: simple rule-based answer
        st.info("AI tidak tersedia. Menjalankan analisis ringkas otomatis (fallback).")
        # crude rules: check latest ratios
        latest = plot_df.iloc[-1]
        bullets = []
        if not np.isnan(latest.get("Current Ratio", np.nan)):
            cr = latest["Current Ratio"]
            bullets.append(f"Current Ratio terakhir: {cr:.2f} — {'cukup baik' if cr>=1 else 'risiko likuiditas'}")
        if not np.isnan(latest.get("Debt to Equity", np.nan)):
            der = latest["Debt to Equity"]
            bullets.append(f"Debt-to-Equity terakhir: {der:.2f} — {'aman' if der<1.5 else 'tinggi leverage'}")
        if not np.isnan(latest.get("ROA (%)", np.nan)):
            roa = latest["ROA (%)"]
            bullets.append(f"ROA: {roa:.2f}% — {'baik' if roa>5 else 'perlu perbaikan margin'}")
        st.write("**Analisis singkat (fallback):**")
        for b in bullets:
            st.write("- " + b)
        st.session_state.chat_history.append({"role": "assistant", "content": "Analisis ringkas fallback: " + " | ".join(bullets)})

# Display chat history
if st.session_state.chat_history:
    st.markdown("**Riwayat Chat**")
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.markdown(f"**👤 Anda:** {m['content']}")
        else:
            st.markdown(f"**🤖 Copilot:** {m['content']}")
