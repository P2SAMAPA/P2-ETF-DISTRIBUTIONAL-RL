# Distributional Reinforcement Learning Engine

Implements C51 (Categorical DQN) to learn the full distribution of next‑day ETF returns, not just the expected value. Uses the Conditional Value at Risk (CVaR) at the 5th percentile as the risk‑sensitive score. Higher CVaR indicates better downside protection. Multi‑window evaluation selects the best window per ETF.

- **Algorithm:** Categorical DQN (Bellemare et al., 2017)
- **Atoms:** 51 spanning [-0.05, 0.05]
- **Risk measure:** CVaR (α = 0.05)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by CVaR

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
