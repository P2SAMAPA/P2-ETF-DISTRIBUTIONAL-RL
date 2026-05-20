import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Distributional RL", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Distributional Reinforcement Learning Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">C51 Categorical DQN | Full return distributions | CVaR risk measure | Multi‑window selection</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 📊 Distributional RL")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Atoms:** {config.N_ATOMS} | **CVaR α:** {config.CVaR_ALPHA}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'distributional_rl_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Conditional Value at Risk (CVaR) – Higher is Better")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - **Distributional RL** learns the full probability distribution of next‑day returns for each ETF, not just the mean.
    - The model uses **C51** (Categorical DQN) with 51 atoms spanning the return support.
    - **CVaR (Conditional Value at Risk)** at α = 5% gives the average return in the worst 5% of the distribution.
    - A higher CVaR means the ETF has less downside risk in the left tail → more attractive for risk‑averse investors.
    - For each ETF, the rolling window that gives the **highest CVaR** is selected.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">CVaR = {etf['cvar']:.6f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    score = info.get("score", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    score = info
                    win = "N/A"
                rows.append({"ETF": ticker, "CVaR": score, "Best Window": win})
            df = pd.DataFrame(rows)
            df["CVaR"] = pd.to_numeric(df["CVaR"], errors='coerce')
            df = df.dropna(subset=["CVaR"]).sort_values("CVaR", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("C51 distributional RL learns the full return distribution. CVaR (α=0.05) is the expected return in the worst 5% of scenarios – a risk‑adjusted measure. Higher CVaR = better downside protection → overweight signal.")
