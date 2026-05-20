import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from distributional_rl import train_c51, predict_cvar

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj

def create_features(returns_df, etf, window, seq_len=5):
    """
    Create features from recent returns of the ETF.
    """
    ret = returns_df[etf].iloc[-window:].copy()
    if len(ret) < seq_len + 1:
        return None, None
    X, y = [], []
    for i in range(seq_len, len(ret)-1):
        X.append(ret.iloc[i-seq_len:i].values)
        y.append(ret.iloc[i+1])
    return np.array(X), np.array(y)

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Distributional RL) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_scores = {}
            for etf in tickers:
                if etf not in returns.columns:
                    continue
                X, y = create_features(returns, etf, win, seq_len=5)
                if X is None or len(X) < 20:
                    continue
                # Train C51 model on this ETF for this window
                model = train_c51(X, y, input_dim=X.shape[1],
                                  n_atoms=config.N_ATOMS,
                                  v_min=config.V_MIN,
                                  v_max=config.V_MAX,
                                  hidden_dim=config.HIDDEN_DIM,
                                  lr=config.LEARNING_RATE,
                                  epochs=config.EPOCHS,
                                  batch_size=config.BATCH_SIZE,
                                  device=device)
                # Predict CVaR for the most recent input vector
                last_X = X[-1:].reshape(1, -1)
                cvar = predict_cvar(model, last_X, alpha=config.CVaR_ALPHA)
                etf_scores[etf] = cvar
            window_results[win] = etf_scores
            for etf, score in etf_scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "cvar": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by CVaR: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/distributional_rl_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Distributional RL Engine complete ===")

if __name__ == "__main__":
    main()
