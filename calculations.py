from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Internal helper ───────────────────────────────────────────────────────────

def _portfolio_daily_returns(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> pd.Series | None:
    """Value-weighted daily return series. Shared by all risk metric functions."""
    if price_df.empty or positions_df.empty:
        return None
    returns = price_df.pct_change().dropna()
    tickers = [t for t in positions_df["Ticker"].tolist() if t in returns.columns]
    if not tickers:
        return None
    weights = positions_df.set_index("Ticker").loc[tickers, "Weight"] / 100
    return returns[tickers].dot(weights)


# ── Sector mapping for ETFs ───────────────────────────────────────────────────

ETF_SECTOR_MAP: dict[str, str] = {
    "VOO": "Broad US Equity",    "VTI": "Broad US Equity",
    "SPY": "Broad US Equity",    "IVV": "Broad US Equity",
    "QQQM": "Information Technology", "QQQ": "Information Technology",
    "XLK": "Information Technology",  "VGT": "Information Technology",
    "VDC": "Consumer Staples",   "XLP": "Consumer Staples",
    "XLF": "Financials",         "VFH": "Financials",
    "XLE": "Energy",             "VDE": "Energy",
    "XLV": "Health Care",        "VHT": "Health Care",
    "XLI": "Industrials",        "VIS": "Industrials",
    "VFMO": "Factor / Other",    "MTUM": "Factor / Other",
    "BTC":  "Digital Assets",    "GBTC": "Digital Assets",
    "ETH":  "Digital Assets",    "ETHE": "Digital Assets",
    "ACWI": "International",     "URTH": "International",
    "VEA":  "International",     "EFA":  "International",
    "EEM":  "Emerging Markets",  "VWO":  "Emerging Markets",
    "TLT":  "Fixed Income",      "AGG":  "Fixed Income",
    "BND":  "Fixed Income",      "SHY":  "Fixed Income",
    "GLD":  "Commodities",       "IAU":  "Commodities",
    "BPTRX": "Multi-Cap Growth", "ARKK": "Innovation",
}

# Common individual stock sector classifications (fallback when yfinance is rate-limited)
STOCK_SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "ORCL": "Information Technology", "INTC": "Information Technology",
    "AMD":  "Information Technology", "QCOM": "Information Technology",
    "CRM":  "Information Technology", "NOW":  "Information Technology",
    "ADBE": "Information Technology", "TXN":  "Information Technology",
    "PLTR": "Information Technology", "SNOW": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META":  "Communication Services", "NFLX": "Communication Services",
    "DIS":   "Communication Services", "CMCSA": "Communication Services",
    "T":     "Communication Services", "VZ":   "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "NKE":  "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "MCD":  "Consumer Discretionary", "HD":   "Consumer Discretionary",
    "LOW":  "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "DKNG": "Consumer Discretionary",
    # Consumer Staples
    "WMT":  "Consumer Staples", "COST": "Consumer Staples",
    "PG":   "Consumer Staples", "KO":   "Consumer Staples",
    "PEP":  "Consumer Staples", "PM":   "Consumer Staples",
    "MDLZ": "Consumer Staples", "CL":   "Consumer Staples",
    # Financials
    "JPM":   "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS":    "Financials", "MS":  "Financials", "V":   "Financials",
    "MA":    "Financials", "AXP": "Financials", "C":   "Financials",
    "BRK-B": "Financials", "SCHW": "Financials", "BX": "Financials",
    "PYPL":  "Financials",
    # Health Care
    "LLY":  "Health Care", "UNH": "Health Care", "JNJ": "Health Care",
    "ABBV": "Health Care", "MRK": "Health Care", "TMO": "Health Care",
    "ABT":  "Health Care", "PFE": "Health Care", "AMGN": "Health Care",
    "GILD": "Health Care", "BMY": "Health Care", "ISRG": "Health Care",
    # Industrials
    "CAT": "Industrials", "GE":  "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "BA":  "Industrials", "RTX": "Industrials",
    "DE":  "Industrials", "LMT": "Industrials", "GD":  "Industrials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "OXY": "Energy",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AEP": "Utilities", "D":   "Utilities",
    # Real Estate
    "PLD":  "Real Estate", "AMT":  "Real Estate", "EQIX": "Real Estate",
    "SPG":  "Real Estate", "O":    "Real Estate",
    # Materials
    "LIN":  "Materials", "APD": "Materials", "NEM": "Materials",
    "FCX":  "Materials", "DOW": "Materials",
}

# S&P 500 sector weights (GICS, approximate as of mid-2025)
SP500_SECTOR_WEIGHTS: dict[str, float] = {
    "Information Technology": 31.4,
    "Financials":             13.1,
    "Health Care":            12.1,
    "Consumer Discretionary": 10.7,
    "Industrials":             8.9,
    "Communication Services":  8.7,
    "Consumer Staples":        5.8,
    "Energy":                  3.7,
    "Real Estate":             2.4,
    "Utilities":               2.3,
    "Materials":               2.2,
}


# ── Position building ─────────────────────────────────────────────────────────

def build_positions(
    portfolio_df: pd.DataFrame,
    ticker_info: dict,
    computed_betas: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for _, row in portfolio_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        try:
            shares = float(row["Shares"])
            avg_cost = float(row["Avg_Cost"])
        except (ValueError, TypeError):
            continue

        info = ticker_info.get(ticker, {})
        current_price = info.get("current_price")
        if current_price is None:
            continue

        value = shares * current_price
        cost_basis = shares * avg_cost
        pnl = value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
        # Prefer beta computed from this window's actual price history vs the
        # benchmark; fall back to Yahoo's figure. If neither exists leave NaN —
        # assuming market beta badly distorts bonds/gold/crypto positions.
        raw_beta = (computed_betas or {}).get(ticker, info.get("beta"))
        try:
            beta = float(raw_beta) if raw_beta is not None else float("nan")
        except (TypeError, ValueError):
            beta = float("nan")
        sector = (ETF_SECTOR_MAP.get(ticker)
                  or STOCK_SECTOR_MAP.get(ticker)
                  or info.get("sector")
                  or "Other")

        rows.append({
            "Ticker":        ticker,
            "Shares":        shares,
            "Avg_Cost":      avg_cost,
            "Current_Price": current_price,
            "Value":         value,
            "Cost_Basis":    cost_basis,
            "PnL":           pnl,
            "PnL_Pct":       pnl_pct,
            "Beta":          beta,
            "Market_Cap":    info.get("market_cap"),
            "Sector":        sector,
            "Dividend_Yield": info.get("dividend_yield"),
            "Week52High":    info.get("week_52_high"),
            "Week52Low":     info.get("week_52_low"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_value = df["Value"].sum()
    df["Weight"] = (df["Value"] / total_value * 100).round(4)
    df["Beta_Contribution"] = (df["Beta"] * df["Weight"] / 100).round(4)
    return df.reset_index(drop=True)


# ── Core portfolio metrics ────────────────────────────────────────────────────

def calc_portfolio_beta(positions_df: pd.DataFrame) -> float:
    """Weighted-average beta over positions with a known beta (renormalized)."""
    known = positions_df.dropna(subset=["Beta"])
    w = float(known["Weight"].sum()) if not known.empty else 0.0
    if w == 0:
        return 0.0
    return float((known["Beta"] * known["Weight"]).sum() / w)


def calc_ticker_betas(
    price_df: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    min_obs: int = 60,
) -> dict[str, float]:
    """Per-ticker beta vs the benchmark, computed from fetched price history."""
    betas: dict[str, float] = {}
    if price_df.empty or benchmark_returns is None or benchmark_returns.empty:
        return betas
    rets = price_df.pct_change()
    for t in rets.columns:
        aligned = pd.DataFrame({"a": rets[t], "b": benchmark_returns}).dropna()
        if len(aligned) < min_obs:
            continue
        bvar = float(aligned["b"].var())
        if bvar > 0:
            betas[t] = float(aligned["a"].cov(aligned["b"]) / bvar)
    return betas


def calc_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty or price_df.shape[1] < 2:
        return pd.DataFrame()
    return price_df.pct_change().dropna().corr()


def calc_portfolio_cumulative(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> pd.Series | None:
    """Cumulative return series for constant-weight backtest, starting at 1.0."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return None
    return (1 + port_returns).cumprod()


def calc_hhi(positions_df: pd.DataFrame) -> float:
    weights = positions_df["Weight"] / 100
    return float((weights ** 2).sum())


# ── Return distribution / tail risk ──────────────────────────────────────────

def calc_var(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    confidence: float = 0.95,
    lookback: int | None = None,
) -> dict:
    empty = {"var_1d": None, "var_5d": None, "var_1d_pct": None, "var_5d_pct": None}
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return empty
    if lookback:
        port_returns = port_returns.tail(lookback)
    if len(port_returns) < 30:
        return empty
    var_1d_pct = float(np.percentile(port_returns, (1 - confidence) * 100))
    var_5d_pct = var_1d_pct * np.sqrt(5)
    total_value = float(positions_df["Value"].sum())
    return {
        "var_1d":     abs(var_1d_pct) * total_value,
        "var_5d":     abs(var_5d_pct) * total_value,
        "var_1d_pct": abs(var_1d_pct) * 100,
        "var_5d_pct": abs(var_5d_pct) * 100,
    }


def calc_cvar(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    confidence: float = 0.95,
    lookback: int | None = None,
) -> dict:
    """Expected Shortfall: average loss beyond the VaR threshold."""
    empty = {"cvar_1d": None, "cvar_1d_pct": None}
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return empty
    if lookback:
        port_returns = port_returns.tail(lookback)
    if len(port_returns) < 30:
        return empty
    threshold = np.percentile(port_returns, (1 - confidence) * 100)
    tail = port_returns[port_returns <= threshold]
    if tail.empty:
        return empty
    cvar_pct = float(tail.mean())
    total_value = float(positions_df["Value"].sum())
    return {
        "cvar_1d":     abs(cvar_pct) * total_value,
        "cvar_1d_pct": abs(cvar_pct) * 100,
    }


def calc_var_multi_confidence(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    lookback: int | None = None,
) -> dict | None:
    """VaR and CVaR at 90%, 95%, 99% confidence levels."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    if lookback:
        port_returns = port_returns.tail(lookback)
    total_value = float(positions_df["Value"].sum())
    result: dict = {}
    for conf in [0.90, 0.95, 0.99]:
        pct_key = int(conf * 100)
        threshold = float(np.percentile(port_returns, (1 - conf) * 100))
        tail = port_returns[port_returns <= threshold]
        result[f"var_{pct_key}"]  = abs(threshold) * total_value
        result[f"cvar_{pct_key}"] = abs(float(tail.mean())) * total_value if not tail.empty else None
    return result


def calc_position_var(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    confidence: float = 0.95,
    lookback: int | None = None,
) -> dict:
    """1-Day VaR per individual position."""
    if price_df.empty or positions_df.empty:
        return {}
    returns = price_df.pct_change().dropna()
    if lookback:
        returns = returns.tail(lookback)
    result: dict = {}
    for _, row in positions_df.iterrows():
        t = row["Ticker"]
        if t not in returns.columns:
            continue
        col = returns[t].dropna()
        if len(col) < 30:
            continue
        var_pct = float(np.percentile(col, (1 - confidence) * 100))
        result[t] = abs(var_pct) * float(row["Value"])
    return result


def calc_marginal_risk_contribution(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    confidence: float = 0.95,
    lookback: int | None = None,
) -> dict:
    """% of total portfolio risk attributed to each position (marginal CVaR)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return {}

    port_tail_window = port_returns.tail(lookback) if lookback else port_returns
    threshold = float(np.percentile(port_tail_window, (1 - confidence) * 100))

    returns = price_df.pct_change().dropna()
    if lookback:
        returns = returns.tail(lookback)
    tickers = [t for t in positions_df["Ticker"].tolist() if t in returns.columns]
    if not tickers:
        return {}

    weights = positions_df.set_index("Ticker").loc[tickers, "Weight"] / 100
    common = port_tail_window.index.intersection(returns.index)
    if len(common) < 10:
        return {}

    port_common = port_tail_window.reindex(common)
    tail_mask = port_common <= threshold
    tail_ret = returns.loc[common][tickers][tail_mask]
    if tail_ret.empty:
        return {}

    port_cvar = abs(float(port_common[tail_mask].mean()))
    if port_cvar == 0:
        return {}

    marginal: dict = {}
    for t in tickers:
        w = float(weights[t])
        pos_mean = float(tail_ret[t].mean())
        marginal[t] = abs(w * pos_mean) / port_cvar

    total = sum(marginal.values())
    if total > 0:
        marginal = {k: v / total * 100 for k, v in marginal.items()}
    return marginal


def calc_max_drawdown(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> tuple[float | None, pd.Series | None]:
    """Returns (max drawdown fraction, full drawdown time series)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 2:
        return None, None
    cum = (1 + port_returns).cumprod()
    drawdown = cum / cum.expanding().max() - 1
    return float(drawdown.min()), drawdown


def calc_annualized_volatility(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> float | None:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 2:
        return None
    return float(port_returns.std() * np.sqrt(252))


def calc_ulcer_index(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    lookback: int | None = None,
) -> float | None:
    """sqrt(mean(drawdown_pct^2)). Measures depth + duration of drawdowns."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    pr = port_returns.tail(lookback) if lookback else port_returns
    cum = (1 + pr).cumprod()
    dd_pct = (cum / cum.expanding().max() - 1) * 100
    return float(np.sqrt((dd_pct ** 2).mean()))


def calc_pain_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
    lookback: int | None = None,
) -> float | None:
    """(Annualized Return - RFR) / Ulcer Index."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    pr = port_returns.tail(lookback) if lookback else port_returns
    ulcer = calc_ulcer_index(price_df, positions_df, lookback)
    if ulcer is None or ulcer == 0:
        return None
    n_years = max(len(pr) / 252, 0.01)
    ann_return = float((1 + pr).prod() ** (1 / n_years) - 1)
    return float((ann_return - risk_free_rate) / ulcer)


def calc_omega_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    threshold: float = 0.0,
    lookback: int | None = None,
) -> float | None:
    """Probability-weighted ratio of gains to losses relative to threshold."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    pr = port_returns.tail(lookback) if lookback else port_returns
    daily_thresh = threshold / 252
    gains  = (pr[pr > daily_thresh]  - daily_thresh).sum()
    losses = (daily_thresh - pr[pr <= daily_thresh]).sum()
    return float(gains / losses) if losses > 0 else None


# ── Risk-adjusted return ratios ───────────────────────────────────────────────

def calc_sharpe_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> float | None:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    excess = port_returns - risk_free_rate / 252
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(252)) if std else None


def calc_sortino_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> float | None:
    """(Ann. excess return) / (ann. downside deviation)."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    daily_mar = risk_free_rate / 252
    n_years = max(len(port_returns) / 252, 1e-9)
    ann_return = float((1 + port_returns).prod() ** (1 / n_years) - 1)   # geometric
    # Standard Sortino: downside deviation over ALL observations, min(r - MAR, 0);
    # dividing by only the downside-day count overstates it (understates the ratio).
    downside = (port_returns - daily_mar).clip(upper=0)
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))
    return float((ann_return - risk_free_rate) / downside_dev) if downside_dev else None


def calc_calmar_ratio(
    price_df: pd.DataFrame, positions_df: pd.DataFrame
) -> float | None:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    n_years = len(port_returns) / 252
    total_return = (1 + port_returns).prod() - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    cum = (1 + port_returns).cumprod()
    max_dd = float((cum / cum.expanding().max() - 1).min())
    return float(cagr / abs(max_dd)) if max_dd != 0 else None


def calc_treynor_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    port_beta: float,
    risk_free_rate: float = 0.05,
) -> float | None:
    if port_beta == 0:
        return None
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None
    n_years = max(len(port_returns) / 252, 1e-9)
    ann_return = float((1 + port_returns).prod() ** (1 / n_years) - 1)   # geometric
    return float((ann_return - risk_free_rate) / port_beta)


# ── Benchmark-relative metrics ────────────────────────────────────────────────

def calc_alpha_r2_tracking_error(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> dict:
    empty = {"alpha": None, "r2": None, "tracking_error": None}
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return empty
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 30:
        return empty
    daily_rf = risk_free_rate / 252
    _, intercept, r_value, _, _ = stats.linregress(
        aligned["bench"] - daily_rf, aligned["port"] - daily_rf
    )
    alpha_ann = (1 + intercept) ** 252 - 1
    tracking_error = (aligned["port"] - aligned["bench"]).std() * np.sqrt(252)
    return {
        "alpha":         float(alpha_ann),
        "r2":            float(r_value ** 2),
        "tracking_error": float(tracking_error),
    }


def calc_capture_ratios(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> tuple[float | None, float | None]:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return None, None
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    up   = aligned[aligned["bench"] > 0]
    down = aligned[aligned["bench"] < 0]

    def geo_ann(series: pd.Series, n: int) -> float:
        return (1 + series).prod() ** (252 / n) - 1

    up_capture = None
    if len(up) >= 10:
        bench_up = geo_ann(up["bench"], len(up))
        if bench_up != 0:
            up_capture = geo_ann(up["port"], len(up)) / bench_up * 100

    down_capture = None
    if len(down) >= 10:
        bench_down = geo_ann(down["bench"], len(down))
        if bench_down != 0:
            down_capture = geo_ann(down["port"], len(down)) / bench_down * 100

    return up_capture, down_capture


def calc_information_ratio(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> float | None:
    """Annualized active return / tracking error."""
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return None
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < 30:
        return None
    active = aligned["port"] - aligned["bench"]
    te = float(active.std() * np.sqrt(252))
    if te == 0:
        return None
    return float(active.mean() * 252 / te)


# ── Rolling metrics ───────────────────────────────────────────────────────────

def calc_rolling_metrics(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    window: int = 63,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < window + 5:
        return pd.DataFrame()
    daily_rf = risk_free_rate / 252
    roll_std = port_returns.rolling(window).std()
    roll_sharpe = (port_returns.rolling(window).mean() - daily_rf) / roll_std * np.sqrt(252)
    roll_vol = roll_std * np.sqrt(252) * 100
    return pd.DataFrame({
        "Rolling Sharpe":        roll_sharpe,
        "Rolling Volatility (%)": roll_vol,
    }).dropna()


def calc_rolling_beta(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 90,
) -> pd.Series:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return pd.Series(dtype=float)
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < window + 5:
        return pd.Series(dtype=float)
    cov = aligned["port"].rolling(window).cov(aligned["bench"])
    var = aligned["bench"].rolling(window).var()
    return (cov / var).dropna()


def calc_rolling_correlation(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 90,
) -> pd.Series:
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None:
        return pd.Series(dtype=float)
    aligned = pd.DataFrame({"port": port_returns, "bench": benchmark_returns}).dropna()
    if len(aligned) < window + 5:
        return pd.Series(dtype=float)
    return aligned["port"].rolling(window).corr(aligned["bench"]).dropna()


# ── Composite scores ──────────────────────────────────────────────────────────

def calc_diversification_score(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
) -> float | None:
    """Score 1–10: higher = more diversified (low avg correlation, more holdings)."""
    if price_df.empty or positions_df.empty:
        return None
    tickers = [t for t in positions_df["Ticker"].tolist() if t in price_df.columns]
    if len(tickers) < 2:
        return None
    returns = price_df[tickers].pct_change().dropna()
    corr = returns.corr()
    n = len(tickers)
    upper = corr.values[np.triu_indices(n, k=1)]
    avg_corr = float(np.mean(upper))
    corr_score = max(1.0, min(8.0, (1 - avg_corr) * 4 + 1))
    hold_bonus = min(2.0, n / 5)
    return round(min(10.0, max(1.0, corr_score + hold_bonus)), 1)


def calc_risk_score(
    port_beta: float,
    ann_vol: float | None,
    max_dd: float | None,
    var_pct: float | None,
) -> float | None:
    """Composite risk score 1–10 from beta, vol, max drawdown, and VaR%."""
    if any(v is None for v in [ann_vol, max_dd, var_pct]):
        return None
    beta_score = min(10.0, max(1.0, (port_beta / 2.0) * 9 + 1))
    vol_score  = min(10.0, max(1.0, ann_vol * 100 / 3.0))
    dd_score   = min(10.0, max(1.0, abs(max_dd) * 100 / 5.0))
    var_score  = min(10.0, max(1.0, var_pct / 0.4))
    composite  = beta_score * 0.25 + vol_score * 0.30 + dd_score * 0.25 + var_score * 0.20
    return round(min(10.0, max(1.0, composite)), 1)


# ── Stress tests ──────────────────────────────────────────────────────────────

STRESS_SCENARIOS = [
    ("COVID Crash (Feb–Mar 2020)",     -0.34),
    ("2022 Rate Hike Selloff",          -0.25),
    ("2008 Global Financial Crisis",    -0.57),
    ("Dot-com Bust (2000–02)",          -0.49),
]


def calc_stress_tests(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Estimated portfolio P&L under named historical market shocks (via beta)."""
    if positions_df.empty:
        return pd.DataFrame()
    total_value = float(positions_df["Value"].sum())
    rows = []
    for name, mkt_ret in STRESS_SCENARIOS:
        port_pnl = sum(
            float(r["Beta"]) * mkt_ret * float(r["Value"])
            for _, r in positions_df.iterrows()
            if pd.notna(r["Beta"])   # unknown beta → excluded, not assumed = 1.0
        )
        rows.append({
            "Scenario":               name,
            "Market Shock":           f"{mkt_ret * 100:.0f}%",
            "Est. Portfolio Loss ($)": port_pnl,
            "Est. Portfolio Loss (%)": port_pnl / total_value * 100,
        })
    return pd.DataFrame(rows)


# ── Monte Carlo simulation ────────────────────────────────────────────────────

MC_MAX_ANNUAL_RETURN = 0.15  # Hard ceiling to prevent unrealistic projections


def calc_monte_carlo(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    initial_value: float,
    years: int = 20,
    simulations: int = 1000,
    monthly_contribution: float = 0.0,
    expected_annual_return: float | None = None,
    inflation_rate: float = 0.03,
) -> dict | None:
    """
    Returns dict with:
      df                  — nominal percentile paths (year, p5, p25, p50, p75, p95)
      df_real             — inflation-adjusted paths
      historical_ann_return — raw trailing annualised return
      used_ann_return     — actual return used in simulation (capped / overridden)
      capped              — True if raw history exceeded MC_MAX_ANNUAL_RETURN
    """
    port_returns = _portfolio_daily_returns(price_df, positions_df)
    if port_returns is None or len(port_returns) < 30:
        return None

    hist_daily_mean  = float(port_returns.mean())
    hist_ann_return  = (1 + hist_daily_mean) ** 252 - 1
    daily_std        = float(port_returns.std())

    if expected_annual_return is not None:
        used_ann_return = min(float(expected_annual_return), MC_MAX_ANNUAL_RETURN)
    else:
        used_ann_return = min(hist_ann_return, MC_MAX_ANNUAL_RETURN)
    used_ann_return  = max(used_ann_return, -0.10)
    used_daily_mean  = (1 + used_ann_return) ** (1 / 252) - 1

    n_days = years * 252
    rng = np.random.default_rng(42)
    rand_rets = rng.normal(used_daily_mean, daily_std, (n_days, simulations))

    paths = np.empty((n_days + 1, simulations))
    paths[0] = initial_value
    monthly_days = 21
    for t in range(1, n_days + 1):
        paths[t] = paths[t - 1] * (1 + rand_rets[t - 1])
        if monthly_contribution > 0 and t % monthly_days == 0:
            paths[t] += monthly_contribution

    pcts = [5, 25, 50, 75, 95]
    data = np.percentile(paths, pcts, axis=1).T
    df = pd.DataFrame(data, columns=[f"p{p}" for p in pcts])
    df["year"] = np.linspace(0, years, n_days + 1)

    # Inflation-adjusted (real) paths
    deflator = (1 + inflation_rate) ** np.linspace(0, years, n_days + 1)
    df_real = df.copy()
    for col in [f"p{p}" for p in pcts]:
        df_real[col] = df_real[col] / deflator

    return {
        "df":                    df,
        "df_real":               df_real,
        "historical_ann_return": hist_ann_return,
        "used_ann_return":       used_ann_return,
        "capped":                hist_ann_return > MC_MAX_ANNUAL_RETURN and expected_annual_return is None,
    }
