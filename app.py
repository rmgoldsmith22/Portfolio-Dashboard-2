from __future__ import annotations

import base64
import collections
import html
import io
import math
import pathlib

import numpy as np
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from calculations import (
    SP500_SECTOR_WEIGHTS,
    build_positions,
    calc_alpha_r2_tracking_error,
    calc_annualized_volatility,
    calc_calmar_ratio,
    calc_capture_ratios,
    calc_correlation_matrix,
    calc_cvar,
    calc_diversification_score,
    calc_hhi,
    calc_information_ratio,
    calc_marginal_risk_contribution,
    calc_max_drawdown,
    calc_monte_carlo,
    calc_omega_ratio,
    calc_pain_ratio,
    calc_portfolio_beta,
    calc_portfolio_cumulative,
    calc_position_var,
    calc_risk_score,
    calc_rolling_beta,
    calc_rolling_correlation,
    calc_rolling_metrics,
    calc_sharpe_ratio,
    calc_sortino_ratio,
    calc_stress_tests,
    calc_ticker_betas,
    calc_treynor_ratio,
    calc_ulcer_index,
    calc_var,
    calc_var_multi_confidence,
)
from data_fetcher import fetch_price_history, fetch_ticker_info

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Risk Dashboard", page_icon=None, layout="wide")

# ── "Official" light palette (white bg, black ink, Libre Franklin) ───────────
# Color is reserved for red / green / yellow status indicators only — all other
# UI chrome stays black / white / neutral gray.

INK         = "#1a1a1a"   # primary text + primary chart data marks
APPLE_GREEN = "#1a7a3c"   # positive / low-risk indicator
APPLE_RED   = "#c0392b"   # negative / high-risk indicator
YELLOW      = "#b8860b"   # caution indicator (amber-gold, readable on white)
APPLE_GRAY  = "#6b6b66"   # secondary / muted label text
APPLE_WHITE = INK         # name retained: primary ink for text + data marks
APPLE_BLUE  = "#9a9a94"   # benchmark series — neutral gray, not a real accent
NAVY_BG     = "#ffffff"   # page background
PANEL_BG    = "#ffffff"   # card surface
SUBTLE      = "#e8e8e4"   # chart gridlines / dividers / zero-lines
HAIRLINE    = "rgba(0,0,0,0.12)"  # CSS-only hairline border
INK_SOFT    = "#5f5f5a"   # small uppercase labels (hierarchy below pure black)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Libre Franklin', sans-serif !important; }}
.stApp {{ background-color: {NAVY_BG} !important; }}

/* Headers — official report case, Libre Franklin */
h1, h2, h3,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {{
    font-family: 'Libre Franklin', sans-serif !important;
    color: {INK} !important;
    letter-spacing: -0.005em !important;
    font-weight: 600 !important;
}}
h1 {{ font-size: 22px !important; letter-spacing: -0.01em !important; }}
h2 {{ font-size: 15px !important; }}
/* Section subheaders — restrained uppercase label, institutional feel */
h3 {{ font-size: 11px !important; color: {INK_SOFT} !important; font-weight: 600 !important;
      text-transform: uppercase !important; letter-spacing: 0.1em !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {PANEL_BG} !important;
    border-right: 1px solid {HAIRLINE} !important;
}}
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([class*="material-"]) {{ font-family: 'Libre Franklin', sans-serif !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{ color: {INK} !important; }}

/* Metric cards — flat, hairline border, no rounded corners */
[data-testid="metric-container"] {{
    background: {PANEL_BG} !important;
    border: 1px solid {HAIRLINE} !important;
    border-radius: 6px !important;
    padding: 14px 18px !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {APPLE_GRAY} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-weight: 600 !important;
    color: {INK} !important;
    letter-spacing: -0.01em !important;
    font-variant-numeric: tabular-nums !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background-color: transparent !important;
    border-bottom: 1px solid {HAIRLINE} !important;
    gap: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {APPLE_GRAY} !important;
    background-color: transparent !important;
    border: none !important;
    padding: 12px 18px !important;
}}
.stTabs [aria-selected="true"] {{
    color: {INK} !important;
    border-bottom: 2px solid {INK} !important;
    background-color: transparent !important;
}}

/* Expander */
[data-testid="stExpander"] {{
    background: {PANEL_BG} !important;
    border: 1px solid {HAIRLINE} !important;
    border-radius: 6px !important;
}}
[data-testid="stExpander"] summary {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-weight: 500 !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {APPLE_GRAY} !important;
}}

/* Input labels */
.stSlider label, .stNumberInput label, .stSelectbox label,
.stFileUploader label, .stRadio label, .stTextInput label {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: {APPLE_GRAY} !important;
}}

/* Buttons */
.stButton > button {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-weight: 500 !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    background-color: {PANEL_BG} !important;
    color: {INK} !important;
    border: 1px solid {HAIRLINE} !important;
    border-radius: 6px !important;
    padding: 8px 14px !important;
}}
.stButton > button:hover {{
    background-color: #f4f4f1 !important;
    border-color: {INK} !important;
    color: {INK} !important;
}}

/* Inputs */
input, textarea, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {{
    font-family: 'Libre Franklin', sans-serif !important;
    background-color: {PANEL_BG} !important;
    border: 1px solid {HAIRLINE} !important;
    border-radius: 6px !important;
    color: {APPLE_WHITE} !important;
}}

[data-testid="stDataFrame"], [data-testid="stTable"] {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-size: 12px !important;
}}

/* Body text → Libre Franklin, but DO NOT touch Material icon spans (their
   font-family carries the glyph; overriding it prints the raw ligature text) */
p, li, div,
span:not([data-testid="stIconMaterial"]):not([class*="material-"]) {{
    font-family: 'Libre Franklin', sans-serif !important;
}}

/* Force readable black body text even if Streamlit's base theme is dark.
   Scoped to containers only: this sets an INHERITED value, so any element with
   its own color (badges, deltas, indicators) still wins. */
.stApp, [data-testid="stMarkdownContainer"] {{
    color: {INK} !important;
}}
h4, h5, h6 {{ color: {INK} !important; font-family: 'Libre Franklin', sans-serif !important;
              font-weight: 600 !important; letter-spacing: -0.005em !important; }}
h4 {{ font-size: 14px !important; }}

/* Restore Material icon fonts (the broad rule above must not clobber them) */
span[data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
.material-symbols-rounded, .material-symbols-outlined {{
    font-family: 'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
}}

/* Light surfaces for uploader / inputs regardless of base theme */
[data-testid="stFileUploaderDropzone"] {{
    background: #f4f4f1 !important;
    border: 1px dashed {HAIRLINE} !important;
}}
[data-testid="stFileUploaderDropzone"] span:not([data-testid="stIconMaterial"]):not([class*="material-"]),
[data-testid="stFileUploaderDropzone"] small {{ color: {INK_SOFT} !important; }}

/* Captions — quiet, readable, normal case */
[data-testid="stCaption"], [data-testid="stCaption"] p {{
    color: {APPLE_GRAY} !important;
    font-size: 11px !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}}

/* Suppress input keyboard artifacts */
[data-testid="InputInstructions"] {{ display: none !important; }}

/* Dividers — hairline only */
hr {{ border-color: {HAIRLINE} !important; border-top: 1px solid {HAIRLINE} !important; }}

/* ── Period selector pills ──────────────────────────────────────────── */
.period-selector [data-testid="stRadio"] [role="radiogroup"] {{
    gap: 0 !important;
    flex-wrap: wrap !important;
    justify-content: flex-start !important;
}}
.period-selector [data-testid="stRadio"] [role="radiogroup"] label {{
    display: flex !important;
    align-items: center !important;
    background: {PANEL_BG};
    border: 1px solid {HAIRLINE};
    padding: 7px 15px !important;
    margin: 0 -1px 0 0 !important;
    cursor: pointer;
    border-radius: 0;
    transition: all 0.12s ease;
}}
/* Hide the radio input and its circular mark (the child div with no text),
   keep the label text regardless of child order */
.period-selector [data-testid="stRadio"] [role="radiogroup"] label input {{
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
}}
.period-selector [data-testid="stRadio"] [role="radiogroup"] label > div:not(:has(p)) {{
    display: none !important;
}}
.period-selector [data-testid="stRadio"] [role="radiogroup"] label p {{
    font-family: 'Libre Franklin', sans-serif !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    color: {APPLE_GRAY} !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}}
.period-selector [data-testid="stRadio"] [role="radiogroup"] label:hover {{
    border-color: rgba(0,0,0,0.35);
}}
.period-selector [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{
    background: {INK};
    border-color: {INK};
}}
.period-selector [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {{
    color: #ffffff !important;
}}
.period-selector [data-testid="stRadio"] > label {{ display: none !important; }}

/* ── Timeframe label (sits beside the dropdown) ──────────────────────── */
.tf-label {{
    font-family: 'Libre Franklin', sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {INK_SOFT};
    line-height: 38px;   /* vertically centers against the 38px select */
    white-space: nowrap;
}}

/* ── Hero value display ──────────────────────────────────────────── */
.hero-value {{
    font-family: 'Libre Franklin', sans-serif;
    font-size: 38px;
    font-weight: 700;
    color: {INK};
    letter-spacing: -0.02em;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}}
.hero-label {{
    font-family: 'Libre Franklin', sans-serif;
    font-size: 10px;
    font-weight: 500;
    color: {APPLE_GRAY};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}}
.hero-delta-pos {{ color: {APPLE_GREEN}; font-weight: 500; font-size: 12px; letter-spacing: 0.04em; }}
.hero-delta-neg {{ color: {APPLE_RED};   font-weight: 500; font-size: 12px; letter-spacing: 0.04em; }}
.hero-delta-neu {{ color: {APPLE_GRAY};  font-weight: 500; font-size: 12px; letter-spacing: 0.04em; }}

/* ── Inline badges ──────────────────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border: 1px solid transparent;
    border-radius: 6px;
    font-family: 'Libre Franklin', sans-serif;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
.badge-red    {{ color: #a32d2d; border-color: #f0d0cc; background: #fbe9e9; }}
.badge-green  {{ color: #1a7a3c; border-color: #cfe8d6; background: #e9f5ed; }}
.badge-yellow {{ color: #8a6500; border-color: #ecdcae; background: #fcf3d9; }}
.badge-gray   {{ color: {APPLE_GRAY}; border-color: {HAIRLINE}; background: #f4f4f1; }}
</style>
""", unsafe_allow_html=True)

HERE = pathlib.Path(__file__).parent

BENCHMARKS: dict[str, str] = {"S&P 500": "^GSPC", "Nasdaq 100": "QQQ", "MSCI World": "ACWI"}
PRIMARY_BENCH = "^GSPC"

# Broadly-diversified index funds — a large weight in one of these is NOT
# single-name concentration risk, so they're excluded from the concentration flag.
BROAD_MARKET_ETFS: set[str] = {
    "VOO", "VTI", "SPY", "IVV", "SPLG", "VV", "SCHB", "SCHX", "ITOT", "IWB",
    "VONE", "SPTM", "FXAIX", "SWPPX", "VFIAX", "VTSAX", "FSKAX", "FZROX",
    "QQQ", "QQQM", "ONEQ", "DIA", "VT", "ACWI", "URTH", "VTWO", "IWM",
    "VXUS", "VEA", "VWO", "SCHD",
}

# ── Time period configuration ─────────────────────────────────────────────────

PERIOD_OPTIONS = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "MAX"]

PERIOD_MAP: dict[str, tuple[str, str]] = {
    # label -> (yfinance period, bar interval)
    "1D":  ("1d",  "1m"),
    "5D":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y":  ("1y",  "1d"),
    "2Y":  ("2y",  "1d"),
    "5Y":  ("5y",  "1d"),
    "10Y": ("10y", "1d"),
    "MAX": ("max", "1d"),
}

# Periods too short for daily-frequency ratios (Sharpe, Sortino, Alpha, etc.)
SHORT_PERIODS = {"1D", "5D", "1M"}

DISCLAIMER = (
    "*For educational purposes only. Not financial advice. Past performance does not guarantee "
    "future results. Consult a qualified financial advisor before making investment decisions.*"
)

# ── Metric help text ──────────────────────────────────────────────────────────

HELP = {
    "beta":           "Beta measures a security's or portfolio's volatility relative to the overall market. The market has a beta of 1.0; a beta above 1.0 means it tends to move more than the market, and below 1.0 less than the market. (Investopedia)",
    "var_1d":         "Value at Risk (VaR) estimates the maximum loss expected over a set time frame at a given confidence level. A 1-day 95% VaR is the loss not expected to be exceeded on about 19 of 20 trading days. (Investopedia)",
    "sharpe":         "The Sharpe ratio is the average return earned in excess of the risk-free rate per unit of total risk (volatility): (Return − Risk-Free Rate) / Standard Deviation. A higher ratio means better risk-adjusted return; above 1.0 is generally considered good. (Investopedia)",
    "sortino":        "Like Sharpe, but penalises only downside moves. Preferred for retirement accounts.",
    "max_dd":         "Maximum drawdown (MDD) is the largest peak-to-trough decline in value before a new peak is reached. It gauges downside risk over a period — the loss of someone who bought at the worst moment. (Investopedia)",
    "hhi":            "The Herfindahl-Hirschman Index (HHI) measures concentration by summing the squares of each holding's portfolio weight. Higher values mean more concentration; under 0.15 is diversified and over 0.25 is concentrated. (Investopedia)",
    "concentration":  "Concentration risk is the potential for loss from holding a large share of the portfolio in a single position. The more weight in one holding, the more company-specific (idiosyncratic) risk you carry. (Investopedia)",
    "cvar":           "Expected Shortfall: average loss on the worst 5% of days.",
    "calmar":         "Annual return / |Max Drawdown|. Above 1.0 means annual gain exceeds the worst historical loss.",
    "treynor":        "Excess return per unit of market risk (beta). Useful when this portfolio is one sleeve of a larger allocation.",
    "ann_vol":        "Annualised standard deviation of daily returns. 15% vol = typically swings ±15% over a year.",
    "alpha":          "Jensen's Alpha: return above CAPM expectation given your beta vs the S&P 500.",
    "r2":             "% of portfolio movement explained by the S&P 500. High R² ≈ index fund.",
    "tracking_error": "Annualised std of (portfolio − benchmark). Low = hugs index; high = active strategy.",
    "up_capture":     "In rising S&P 500 markets, what % of gains did the portfolio capture?",
    "down_capture":   "In falling S&P 500 markets, what % of losses did the portfolio absorb? Lower is better.",
    "rolling":        "Rolling charts reveal whether risk-adjusted performance has been consistent or changed over time.",
    "monte_carlo":    "Simulates thousands of future paths using historical mean return and volatility. A planning range, not a prediction.",
    "ulcer":          "sqrt(mean(drawdown²)). Captures depth and duration. < 5 low; 5–15 moderate; > 15 high.",
    "pain_ratio":     "Annualized return / Ulcer Index. Higher = better reward per unit of drawdown pain.",
    "omega":          "Probability-weighted ratio of gains vs losses above the risk-free rate. > 1.0 means gains outweigh losses.",
    "info_ratio":     "Active return / Tracking Error. > 0.5 = skilled active management.",
    "div_score":      "A composite 1–10 diversification score based on the average pairwise correlation between holdings and how many you hold. Lower correlation and more holdings score higher; a score under 4 flags weak diversification. (Composite metric)",
    "risk_score":     "A composite 1–10 risk score combining portfolio beta (25%), annualized volatility (30%), max drawdown (25%), and 1-day VaR % (20%). Scores above 7 indicate speculative risk levels. (Composite metric)",
}

SECTION_EXPLAINERS = {
    "overview": """
**Risk Score** — Composite 1–10 weighted across annualized volatility (30%), max drawdown (25%), portfolio beta (25%), and 1-day VaR% (20%). Scores above 7 indicate speculative levels.

**Diversification Score** — Based on average pairwise correlation and number of positions. A score below 4 means holdings are highly correlated.

**Portfolio Beta** — Beta 1.2 means a 10% market rally produces a 12% gain (and a 12% loss in a downturn). Values below 1.0 reduce market sensitivity.

**1-Day VaR (95%)** — The dollar loss exceeded no more than once every 20 trading days, computed via historical simulation.

**Sharpe Ratio** — Annualised excess return / annualised total volatility. Above 1.0 is good; long-only equity portfolios typically land 0.3–1.2.

**Sortino Ratio** — Sharpe variant using downside deviation only. More investor-friendly for retirement accounts.

**Max Drawdown** — The steepest valley in portfolio history. Sets realistic expectations for worst-case scenarios.
""",
    "risk": """
**CVaR / Expected Shortfall** — The average loss on the worst 5% of days. Required by Basel III for bank capital.

**Ulcer Index** — Measures depth AND duration of drawdowns. Unlike Max Drawdown, it penalises prolonged underwater periods.

**Pain Ratio** — Annualized return / Ulcer Index. Duration-weighted Calmar.

**Omega Ratio** — All gains vs all losses above the risk-free rate. > 1.0 means cumulative gains dominate.

**Calmar Ratio** — Annual return / |Max Drawdown|. Above 1.0 = earns back worst loss in under a year.

**Treynor Ratio** — Return per unit of systematic (market) risk.

**Stress Tests** — Estimated loss if the S&P 500 reprises a named historical shock, applied through each holding's beta.
""",
    "benchmarks": """
**Jensen's Alpha** — Return above CAPM expectation. Measures manager skill above passive exposure.

**R-Squared** — % of fluctuations explained by the S&P 500. High R² with moderate Sharpe = lots of market risk for limited active return.

**Tracking Error** — Std of (portfolio − benchmark). Index funds run ~0.02%; active managers 3–8%.

**Up/Down Capture** — Ideal: > 100% up-capture and < 100% down-capture.

**Information Ratio** — Active return / Tracking Error. > 0.5 = skilled active management.

**Rolling Beta / Correlation** — Reveal whether market sensitivity has been stable or shifted.
""",
    "monte_carlo": """
**How the simulation works** — Uses a user-specified expected annual return (capped at 15%) and historical volatility. Draws from a normal distribution to generate thousands of independent paths.

**Return cap** — Hard-capped at 15% per year to prevent recent bull-market data from producing unrealistic projections.

**Reading the fan chart:**
- **Median (50th pct)** — Half of paths end above this value.
- **Inner band (25th–75th pct)** — The "most likely" range.
- **Outer band (5th–95th pct)** — Covers 90% of scenarios.

**Real values** — The dashed line deflates the nominal projection by your inflation rate. Represents purchasing power.

**Caveats:**
- Assumes constant weights and that the specified return and historical volatility persist.
- Does not model fat tails, correlation breakdowns, or black swans.
- Does not account for taxes, fees, or dividends unless your inputs reflect them.
""",
    "holdings": """
**Position-level VaR** — 1-day 95% VaR for each holding as a standalone position. Portfolio VaR is lower due to diversification.

**Marginal Risk Contribution** — Share of total portfolio CVaR from each position. > 30% warrants review regardless of P&L.

**52-Week Range** — Where current price sits within the 52-week band.
""",
}

# ── Cached data fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def cached_price_history(tickers_tuple: tuple, period: str, interval: str):
    return fetch_price_history(list(tickers_tuple), period, interval)


@st.cache_data(ttl=300, show_spinner=False)
def cached_ticker_info(ticker: str) -> dict:
    return fetch_ticker_info(ticker)


# ── Company logo + brand-color helpers (allocation chart) ──────────────────────

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 16_000_000  # decompression bombs raise instead of warn
    _HAS_PIL = True
except Exception:  # Pillow not installed → skip color extraction, use fallback
    _HAS_PIL = False

# Curated fallback brand colors for common tickers (used when a logo / its color
# can't be fetched). Keeps the allocation chart colorful even offline.
BRAND_COLORS: dict[str, str] = {
    "AAPL": "#0b0b0b", "MSFT": "#0078D4", "GOOGL": "#4285F4", "GOOG": "#4285F4",
    "AMZN": "#FF9900", "TSLA": "#E82127", "NVDA": "#76B900", "META": "#1877F2",
    "JPM": "#117ACA", "LLY": "#D52B1E", "V": "#1A1F71", "MA": "#EB001B",
    "UNH": "#002677", "HD": "#F96302", "PG": "#003DA5", "KO": "#F40009",
    "PEP": "#004B93", "COST": "#E31837", "DIS": "#113CCF", "NFLX": "#E50914",
    "AMD": "#ED1C24", "INTC": "#0071C5", "CRM": "#00A1E0", "ORCL": "#F80000",
    "ADBE": "#FA0F00", "WMT": "#0071CE", "XOM": "#FF1721", "CVX": "#0066B2",
    "BAC": "#E11B22", "WFC": "#D71E2B", "PFE": "#0093D0", "MRK": "#00857C",
    "ABBV": "#071D49", "T": "#00A8E0", "VZ": "#CD040B", "CSCO": "#1BA0D7",
    "QQQ": "#6f42c1", "SPY": "#1d6fa5", "VOO": "#96281B", "VTI": "#9b1b30",
}
_FALLBACK_BRAND = "#5f6b7a"
_MAX_LOGO_BYTES = 512 * 1024  # cap fetched logo size (memory + page-weight guard)


def _dominant_color(img_bytes: bytes) -> str | None:
    """Most common saturated, non-white/black colour in a logo image."""
    if not _HAS_PIL:
        return None
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        im.thumbnail((64, 64))
        counts: collections.Counter = collections.Counter()
        for r, g, b, a in im.getdata():
            if a < 128:
                continue
            if r > 232 and g > 232 and b > 232:   # near-white
                continue
            if r < 24 and g < 24 and b < 24:        # near-black
                continue
            counts[(r // 24 * 24, g // 24 * 24, b // 24 * 24)] += 1
        if not counts:
            return None
        r, g, b = counts.most_common(1)[0][0]
        return f"#{min(r,255):02x}{min(g,255):02x}{min(b,255):02x}"
    except Exception:
        return None


def _domain_from_website(website: str | None) -> str | None:
    if not website:
        return None
    d = website.replace("https://", "").replace("http://", "").replace("www.", "")
    return d.split("/")[0].strip() or None


@st.cache_data(ttl=86400, show_spinner=False)
def brand_logo_and_color(ticker: str, website: str | None) -> tuple[str | None, str]:
    """Return (logo_data_uri | None, hex_color). Always returns a usable colour.

    Tries a ticker-based logo service first, then the company domain (Clearbit).
    Network/parse failures degrade gracefully to a curated brand colour."""
    fallback = BRAND_COLORS.get(ticker.upper(), _FALLBACK_BRAND)
    domain = _domain_from_website(website)
    sources = [f"https://financialmodelingprep.com/image-stock/{ticker.upper()}.png"]
    if domain:
        sources.append(f"https://logo.clearbit.com/{domain}")

    for url in sources:
        try:
            # No redirects (SSRF guard); streamed with a hard size cap (DoS guard).
            resp = requests.get(url, timeout=4, headers={"User-Agent": "Mozilla/5.0"},
                                allow_redirects=False, stream=True)
            if resp.status_code != 200:
                continue
            ctype = resp.headers.get("Content-Type", "")
            if "image" not in ctype:
                continue
            content = b""
            for chunk in resp.iter_content(64 * 1024):
                content += chunk
                if len(content) > _MAX_LOGO_BYTES:
                    content = b""
                    break
            if len(content) < 200:
                continue
            color = _dominant_color(content) or fallback
            mime = "image/png" if "png" in ctype else ("image/svg+xml" if "svg" in ctype else "image/jpeg")
            if "svg" in mime:   # can't recolor/extract reliably from SVG; use as image only
                color = BRAND_COLORS.get(ticker.upper(), color)
            b64 = base64.b64encode(content).decode("ascii")
            return f"data:{mime};base64,{b64}", color
        except Exception:
            continue
    return None, fallback


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{max(0,min(255,int(r))):02x}{max(0,min(255,int(g))):02x}{max(0,min(255,int(b))):02x}"


def _distinct_color(hex_color: str, used: list[str], thresh: float = 52.0) -> str:
    """Nudge a colour's lightness until it's visibly distinct from ones already
    used (fixes multiple same-brand holdings reading as one shade)."""
    try:
        rgb = _hex_to_rgb(hex_color)
    except Exception:
        return hex_color
    used_rgb = []
    for u in used:
        try:
            used_rgb.append(_hex_to_rgb(u))
        except Exception:
            pass

    def _far_enough(c):
        return all(sum((a - b) ** 2 for a, b in zip(c, u)) ** 0.5 >= thresh for u in used_rgb)

    if _far_enough(rgb):
        return hex_color
    for factor in (0.72, 1.32, 0.55, 1.55, 0.42, 1.75):
        cand = tuple(c * factor for c in rgb)
        if _far_enough(cand):
            return _rgb_to_hex(*cand)
    return hex_color


@st.cache_data(ttl=300, show_spinner=False)
def cached_benchmark_history(period: str, interval: str) -> pd.DataFrame:
    df, _ = fetch_price_history(list(BENCHMARKS.values()), period, interval)
    return df


@st.cache_data(ttl=3600, max_entries=32, show_spinner=False)
def cached_monte_carlo(
    price_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    initial_value: float,
    years: int,
    simulations: int,
    monthly_contribution: float,
    expected_annual_return: float | None,
    inflation_rate: float,
) -> dict | None:
    return calc_monte_carlo(
        price_df, positions_df, initial_value, years, simulations,
        monthly_contribution, expected_annual_return, inflation_rate,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Portfolio Risk")
    st.markdown("---")

    st.subheader("Portfolio Input")
    st.caption("Required columns: `ticker`, `shares`, `avg_cost`")
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"], label_visibility="collapsed",
        help="CSV must have columns: Ticker, Shares, Avg_Cost",
    )
    col_load, col_dl = st.columns(2)
    with col_load:
        load_sample = st.button("Load Sample", use_container_width=True)
    with col_dl:
        st.download_button(
            "Template", use_container_width=True,
            data="Ticker,Shares,Avg_Cost\nAAPL,10,150.00\nMSFT,5,300.00\n",
            file_name="portfolio_template.csv", mime="text/csv",
        )

    if load_sample:
        st.session_state["use_sample"] = True
    if uploaded is not None:
        st.session_state["use_sample"] = False

    st.markdown("---")

    with st.expander("Custom Benchmark Allocation"):
        st.caption("e.g. `SPY:60,AGG:40`  (must sum to 100)")
        custom_alloc_str = st.text_input("Allocation", value="", label_visibility="collapsed",
                                         placeholder="SPY:60,AGG:40")

    st.markdown("---")
    st.subheader("Risk Settings")
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25,
                                help="Annual rate used in Sharpe, Sortino, Treynor, Alpha.") / 100
    mc_sims = st.select_slider("Monte Carlo Paths", options=[500, 1000, 2000, 5000], value=1000)

    st.markdown("---")
    st.subheader("Projection Settings")
    mc_expected_return_pct = st.slider(
        "Expected Annual Return (%)", 1.0, 15.0, 9.0, 0.5,
        help="Override historical mean for Monte Carlo. Capped at 15%.",
    )
    mc_inflation_rate_pct = st.slider(
        "Inflation Rate (%)", 0.0, 8.0, 3.0, 0.5,
        help="Used for inflation-adjusted (real) projected values.",
    )
    mc_use_hist = st.checkbox(
        "Use historical mean return",
        help="Ignore the Expected Return slider and use the portfolio's own "
             "historical mean (still capped at 15%).",
    )

    st.markdown("---")
    st.caption("Data: Yahoo Finance · Cache: 5 min")

# ── Resolve portfolio data ────────────────────────────────────────────────────

st.title("Portfolio Risk Dashboard")
st.caption(
    "Methodology: historical risk metrics apply today's portfolio weights across the "
    "entire lookback window (constant-weight backtest) — they describe the current mix, "
    "not your realized returns."
)

# Timeframe dropdown — label sits to the left of the select
tf_label_col, tf_select_col, _tf_spacer = st.columns([1, 2, 7])
with tf_label_col:
    st.markdown("<div class='tf-label'>Timeframe</div>", unsafe_allow_html=True)
with tf_select_col:
    period_label = st.selectbox(
        "Timeframe", PERIOD_OPTIONS,
        index=6,  # 1Y default
        label_visibility="collapsed",
        key="period_selector",
    )
period, interval = PERIOD_MAP[period_label]
is_short_period = period_label in SHORT_PERIODS

# ── Portfolio CSV normalization (native OR brokerage export) ───────────────────

_TICKER_KEYS   = ("symbol", "ticker", "sym")
_SHARES_KEYS   = ("qty", "quantity", "shares", "units")
_AVGCOST_KEYS  = ("avg cost", "average cost", "cost/share", "cost per share",
                  "avg price", "average price")
_COSTBASIS_KEYS = ("cost basis", "total cost")
_NONPOSITION_RE = r"^(CASH|--)$|TOTAL|ACCOUNT"  # anchored CASH: only exact footer rows match


def _clean_number(x) -> float | None:
    """Turn '$2,336.42', '11.98%', '--', '' into a float or None."""
    if x is None:
        return None
    s = str(x).replace("$", "").replace(",", "").replace("%", "").strip()
    if s.lower() in ("", "--", "n/a", "na", "nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _raw_csv_text(source) -> str:
    if hasattr(source, "getvalue"):
        data = source.getvalue()
        return data.decode("utf-8-sig", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
    with open(source, "r", encoding="utf-8-sig", errors="replace") as fh:
        return fh.read()


def normalize_portfolio_csv(source) -> pd.DataFrame:
    """Parse a portfolio CSV into Ticker / Shares / Avg_Cost.

    Accepts the app's native format (Ticker, Shares, Avg_Cost) and common
    brokerage 'Positions' exports (e.g. Schwab/Fidelity) that have a title row
    before the header, currency-formatted numbers, a total cost-basis column
    instead of per-share cost, and Cash/Total footer rows.
    """
    text = _raw_csv_text(source)
    lines = text.splitlines()

    # Find the header row: first line that has a ticker-like column.
    header_idx = 0
    for i, line in enumerate(lines):
        fields = [f.strip().strip('"').lower().replace("_", " ") for f in line.split(",")]
        if len(fields) >= 2 and any(any(k in f for k in _TICKER_KEYS) for f in fields):
            header_idx = i
            break

    raw = pd.read_csv(io.StringIO(text), skiprows=header_idx, dtype=str, keep_default_na=False)
    low = {c: str(c).strip().strip('"').lower().replace("_", " ") for c in raw.columns}

    def _find(keys):
        for col, name in low.items():
            if any(k in name for k in keys):
                return col
        return None

    tcol, scol = _find(_TICKER_KEYS), _find(_SHARES_KEYS)
    acol, ccol = _find(_AVGCOST_KEYS), _find(_COSTBASIS_KEYS)
    if tcol is None or scol is None:
        return raw  # let downstream validation surface a clear message

    out = pd.DataFrame()
    out["Ticker"] = raw[tcol].astype(str).str.strip().str.upper().str.replace("/", "-", regex=False)
    out["Shares"] = raw[scol].map(_clean_number)
    out["Avg_Cost"] = raw[acol].map(_clean_number) if acol else None

    # Derive per-share cost from a total cost-basis column when needed.
    if ccol is not None:
        basis = raw[ccol].map(_clean_number)
        need = out["Avg_Cost"].isna() & out["Shares"].notna() & (out["Shares"] != 0) & basis.notna()
        out.loc[need, "Avg_Cost"] = basis[need] / out.loc[need, "Shares"]

    # Drop cash / totals / blank rows and anything without usable numbers.
    bad = out["Ticker"].str.contains(_NONPOSITION_RE, case=False, regex=True, na=True)
    # Security: strict ticker whitelist blocks HTML/script payloads at the source
    # (tickers later reach st.markdown(..., unsafe_allow_html=True) render paths).
    bad |= ~out["Ticker"].str.fullmatch(r"[A-Z0-9.\-^=]{1,12}", na=False)
    out = out[~bad].dropna(subset=["Shares", "Avg_Cost"])
    out = out[out["Shares"] != 0]
    out = out.reset_index(drop=True)
    if len(out) > 200:
        raise ValueError(f"{len(out)} positions found — limit is 200 rows.")
    return out


portfolio_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        portfolio_df = normalize_portfolio_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        st.stop()
elif st.session_state.get("use_sample"):
    portfolio_df = normalize_portfolio_csv(HERE / "sample_portfolio.csv")
else:
    st.info("Upload a portfolio CSV or click **Load Sample** in the sidebar to get started.")
    st.stop()

portfolio_df.columns = [c.strip().title().replace(" ", "_") for c in portfolio_df.columns]
missing = {"Ticker", "Shares", "Avg_Cost"} - set(portfolio_df.columns)
if missing:
    st.error(
        f"CSV missing columns: {', '.join(sorted(missing))}. "
        "Expected a native file (Ticker, Shares, Avg_Cost) or a brokerage positions export."
    )
    st.stop()
if portfolio_df.empty:
    st.error("No valid positions found in the CSV after removing cash/total rows.")
    st.stop()

portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.strip().str.upper()
tickers = portfolio_df["Ticker"].tolist()

# ── Fetch market data ─────────────────────────────────────────────────────────

with st.spinner(f"Fetching {period_label} of market data…"):
    price_df, failed = cached_price_history(tuple(tickers), period, interval)
    ticker_info = {t: cached_ticker_info(t) for t in tickers if t not in failed}
    bench_df = cached_benchmark_history(period, interval)

for t in failed:
    st.warning(f"No data for **{t}** — skipped.")

active_df = portfolio_df[~portfolio_df["Ticker"].isin(failed)].copy()
if active_df.empty:
    st.error("No valid tickers. Check your CSV.")
    st.stop()

for t in active_df["Ticker"].tolist():
    if t in price_df.columns:
        last = price_df[t].dropna()
        if not last.empty:
            ticker_info.setdefault(t, {})["current_price"] = float(last.iloc[-1])

_bench_returns: pd.Series | None = None
if not bench_df.empty and PRIMARY_BENCH in bench_df.columns:
    _bench_returns = bench_df[PRIMARY_BENCH].pct_change().dropna()

# Betas from this window's actual price history (fallback inside: Yahoo's beta)
ticker_betas = calc_ticker_betas(price_df, _bench_returns)
positions = build_positions(active_df, ticker_info, ticker_betas)
if positions.empty:
    st.error("Could not build positions — verify tickers have valid prices.")
    st.stop()

valid = positions["Ticker"].tolist()
pdf = price_df[[t for t in valid if t in price_df.columns]]

# ── Custom benchmark allocation ──────────────────────────────────────────────

custom_bench_returns: pd.Series | None = None
if custom_alloc_str.strip():
    try:
        parts = [p.strip() for p in custom_alloc_str.split(",") if p.strip()]
        cb_tickers = []
        cb_weights: list[float] = []
        for part in parts:
            sym, wt = part.split(":")
            cb_tickers.append(sym.strip().upper())
            cb_weights.append(float(wt.strip()) / 100)
        cb_price_df, _ = fetch_price_history(cb_tickers, period, interval)
        if not cb_price_df.empty:
            cb_rets = cb_price_df[[t for t in cb_tickers if t in cb_price_df.columns]].pct_change().dropna()
            avail = [t for t in cb_tickers if t in cb_rets.columns]
            w_norm = np.array([cb_weights[cb_tickers.index(t)] for t in avail])
            w_norm = w_norm / w_norm.sum()
            custom_bench_returns = cb_rets[avail].dot(w_norm)
    except Exception:
        st.sidebar.warning("Could not parse custom allocation — check format (e.g. SPY:60,AGG:40).")

# ── Calculations ──────────────────────────────────────────────────────────────

port_beta = calc_portfolio_beta(positions)
hhi       = calc_hhi(positions)
port_cum  = calc_portfolio_cumulative(pdf, positions)

# Ratios that depend on daily-frequency returns — gated on short periods
if not is_short_period:
    corr             = calc_correlation_matrix(pdf)
    var_data         = calc_var(pdf, positions)
    cvar_data        = calc_cvar(pdf, positions)
    var_multi        = calc_var_multi_confidence(pdf, positions)
    sharpe           = calc_sharpe_ratio(pdf, positions, risk_free_rate)
    sortino          = calc_sortino_ratio(pdf, positions, risk_free_rate)
    calmar           = calc_calmar_ratio(pdf, positions)
    treynor          = calc_treynor_ratio(pdf, positions, port_beta, risk_free_rate)
    max_dd, dd_series = calc_max_drawdown(pdf, positions)
    ann_vol          = calc_annualized_volatility(pdf, positions)
    rolling          = calc_rolling_metrics(pdf, positions, window=63, risk_free_rate=risk_free_rate)
    ulcer            = calc_ulcer_index(pdf, positions)
    pain             = calc_pain_ratio(pdf, positions, risk_free_rate)
    omega            = calc_omega_ratio(pdf, positions, threshold=risk_free_rate)
    div_score        = calc_diversification_score(pdf, positions)
    risk_score_val   = calc_risk_score(port_beta, ann_vol, max_dd, var_data.get("var_1d_pct"))
    stress_df        = calc_stress_tests(positions)
    pos_var          = calc_position_var(pdf, positions)
    mrc              = calc_marginal_risk_contribution(pdf, positions)
else:
    corr = pd.DataFrame()
    var_data = {"var_1d": None, "var_5d": None, "var_1d_pct": None, "var_5d_pct": None}
    cvar_data = {"cvar_1d": None, "cvar_1d_pct": None}
    var_multi = None
    sharpe = sortino = calmar = treynor = ann_vol = ulcer = pain = omega = None
    div_score = risk_score_val = None
    max_dd = None
    dd_series = None
    rolling = pd.DataFrame()
    stress_df = calc_stress_tests(positions)  # OK — uses position-level beta only
    pos_var = {}
    mrc = {}

primary_bench_returns: pd.Series | None = _bench_returns

bench_stats = {"alpha": None, "r2": None, "tracking_error": None}
up_cap = down_cap = info_ratio = None
roll_beta_series = pd.Series(dtype=float)
roll_corr_series = pd.Series(dtype=float)

if (not is_short_period) and primary_bench_returns is not None:
    bench_stats   = calc_alpha_r2_tracking_error(pdf, positions, primary_bench_returns, risk_free_rate)
    up_cap, down_cap = calc_capture_ratios(pdf, positions, primary_bench_returns)
    info_ratio    = calc_information_ratio(pdf, positions, primary_bench_returns)
    roll_beta_series = calc_rolling_beta(pdf, positions, primary_bench_returns)
    roll_corr_series = calc_rolling_correlation(pdf, positions, primary_bench_returns)

total_value   = positions["Value"].sum()
total_cost    = positions["Cost_Basis"].sum()
total_pnl     = positions["PnL"].sum()
total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0.0

lookback_days = len(pdf)

# Period return (for hero number on Overview)
period_return_pct = None
period_return_dollars = None
if port_cum is not None and not port_cum.empty:
    period_return_pct = (port_cum.iloc[-1] / port_cum.iloc[0] - 1) * 100
    period_return_dollars = total_value - (total_value / port_cum.iloc[-1] * port_cum.iloc[0])

# Sidebar risk badge
if risk_score_val is not None:
    score_label_map = [
        (3,  APPLE_GREEN, "LOW RISK"),
        (5,  APPLE_GREEN, "MODERATE"),
        (7,  YELLOW,      "HIGH RISK"),
        (9,  APPLE_RED,   "AGGRESSIVE"),
        (11, APPLE_RED,   "SPECULATIVE"),
    ]
    badge_color, badge_label = APPLE_GRAY, "UNKNOWN"
    for upper, color, lbl in score_label_map:
        if risk_score_val < upper:
            badge_color, badge_label = color, lbl
            break
    # Rendered as a prominent banner at the top of the Overview tab (see below).


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val: float | None, fmt: str, prefix: str = "", suffix: str = "") -> str:
    return f"{prefix}{val:{fmt}}{suffix}" if val is not None else "N/A"


def _dark_chart(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=INK,
        legend=dict(font=dict(color=INK, size=12)),
        margin=dict(t=20, b=20, l=10, r=10),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, color=INK, linecolor=SUBTLE,
                     tickfont=dict(color=INK), title_font=dict(color=INK))
    fig.update_yaxes(gridcolor=SUBTLE, color=INK, linecolor=SUBTLE,
                     tickfont=dict(color=INK), title_font=dict(color=INK))
    return fig


def _short_period_notice():
    st.info(
        f"**Risk-adjusted ratios not available for {period_label}.** "
        f"Sharpe, Sortino, Alpha, R², and capture ratios require at least 3 months of daily data. "
        f"Switch to **3M** or longer to see them."
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_risk, tab_bench, tab_mc, tab_holdings = st.tabs([
    "Overview", "Risk Analysis", "Benchmarks", "Projections", "Holdings"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:

    # ── Portfolio risk banner (prominent, top of page) ───────────────────────
    if risk_score_val is not None:
        _risk_tints = {
            APPLE_GREEN: ("#e9f5ed", "#1a7a3c"),
            YELLOW:      ("#fcf3d9", "#8a6500"),
            APPLE_RED:   ("#fbe9e9", "#a32d2d"),
            APPLE_GRAY:  ("#f4f4f1", "#5f5f5a"),
        }
        _rbg, _rfg = _risk_tints.get(badge_color, ("#f4f4f1", "#5f5f5a"))
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:16px; padding:12px 18px; "
            f"background:{_rbg}; border:1px solid rgba(0,0,0,0.06); border-radius:10px; margin-bottom:16px;'>"
            f"<span style='font-size:11px; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; color:#5f5f5a;'>Portfolio Risk</span>"
            f"<span style='font-size:20px; font-weight:700; color:{_rfg}; letter-spacing:0.03em;'>{badge_label}</span>"
            f"<span style='font-size:13px; color:#5f5f5a; margin-left:auto;'>Risk Score "
            f"<b style='color:#1a1a1a;'>{risk_score_val}/10</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Hero number row ──────────────────────────────────────────────────────
    hero_col1, hero_col2 = st.columns([2, 3])
    with hero_col1:
        st.markdown(f"<div class='hero-label'>Portfolio Value</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hero-value'>${total_value:,.2f}</div>", unsafe_allow_html=True)
        if period_return_pct is not None:
            cls = "hero-delta-pos" if period_return_pct > 0 else ("hero-delta-neg" if period_return_pct < 0 else "hero-delta-neu")
            arrow = "+" if period_return_pct > 0 else ""
            st.markdown(
                f"<div class='{cls}' style='margin-top:8px'>"
                f"{arrow}${period_return_dollars:,.2f}  ·  {arrow}{period_return_pct:.2f}%  "
                f"<span style='color:{APPLE_GRAY}; font-weight:400'>· past {period_label}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<div class='hero-delta-neu'>—</div>", unsafe_allow_html=True)

    with hero_col2:
        st.markdown(f"<div class='hero-label'>Unrealized P&L (since cost basis)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='hero-value'>${total_pnl:,.2f}</div>", unsafe_allow_html=True)
        cls = "hero-delta-pos" if total_pnl_pct > 0 else ("hero-delta-neg" if total_pnl_pct < 0 else "hero-delta-neu")
        st.markdown(f"<div class='{cls}' style='margin-top:8px'>{total_pnl_pct:+.2f}%</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance vs Benchmarks chart ─────────────────────────────────────
    st.subheader(f"Performance vs Benchmarks · {period_label}")
    if port_cum is not None and not port_cum.empty:
        start = port_cum.index[0]
        comparison = pd.DataFrame({"Your Portfolio": port_cum / port_cum.iloc[0] * 100})
        comparison.index = pd.DatetimeIndex(comparison.index)

        bench_label_map = {v: k for k, v in BENCHMARKS.items()}
        for bticker, blabel in bench_label_map.items():
            if bticker in bench_df.columns:
                s = bench_df[bticker].dropna()
                s = s[s.index >= start]
                if not s.empty:
                    s.index = pd.DatetimeIndex(s.index)
                    comparison[blabel] = s / s.iloc[0] * 100

        if custom_bench_returns is not None:
            cb_cum = (1 + custom_bench_returns).cumprod()
            cb_cum = cb_cum[cb_cum.index >= start]
            if not cb_cum.empty:
                cb_cum.index = pd.DatetimeIndex(cb_cum.index)
                comparison["Custom Benchmark"] = cb_cum / cb_cum.iloc[0] * 100

        comparison.index.name = "Date"
        compare_reset = comparison.reset_index()
        fig_cmp = px.line(
            compare_reset, x="Date", y=comparison.columns.tolist(),
            labels={"value": "Growth of $100", "variable": ""},
            color_discrete_map={
                "Your Portfolio":   INK,
                "S&P 500":          "#2a78d6",
                "Nasdaq 100":       "#d97706",
                "MSCI World":       "#7c3aed",
                "Custom Benchmark": "#0d9488",
            },
        )
        # Distinguish each series by BOTH shade and dash pattern so the legend
        # is unambiguous even in a mostly-monochrome palette.
        _line_style = {
            "Your Portfolio":   (INK,       2.8, "solid"),
            "S&P 500":          ("#2a78d6", 1.8, "solid"),
            "Nasdaq 100":       ("#d97706", 1.8, "dash"),
            "MSCI World":       ("#7c3aed", 1.8, "dot"),
            "Custom Benchmark": ("#0d9488", 1.8, "dashdot"),
        }
        for _tr in fig_cmp.data:
            _st = _line_style.get(_tr.name)
            if _st:
                _tr.line.color, _tr.line.width, _tr.line.dash = _st
        _dark_chart(fig_cmp, 380)
        fig_cmp.update_layout(
            yaxis_title="Growth of $100",
            xaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(color=INK, size=12.5),
                        bgcolor="rgba(255,255,255,0.65)", borderwidth=0,
                        title_text=""),
            margin=dict(t=48, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Not enough data to build performance comparison.")

    # ── Risk / Diversification score row ─────────────────────────────────────
    if not is_short_period:
        sc1, sc2, sc3, sc4 = st.columns(4)

        if risk_score_val is not None:
            risk_label = (
                "Low" if risk_score_val < 3 else
                "Moderate" if risk_score_val < 5 else
                "High" if risk_score_val < 7 else
                "Aggressive" if risk_score_val < 9 else "Speculative"
            )
            sc1.metric("Risk Score", f"{risk_score_val}/10", risk_label,
                       delta_color="inverse", help=HELP["risk_score"])
        else:
            sc1.metric("Risk Score", "N/A")

        if div_score is not None:
            div_label = "Low diversification" if div_score < 4 else "Adequate"
            sc2.metric("Diversification Score", f"{div_score}/10", div_label,
                       delta_color="normal", help=HELP["div_score"])
        else:
            sc2.metric("Diversification Score", "N/A")

        # Concentration = largest SINGLE-NAME position. Broad-market index funds
        # (e.g. VOO, VTI, QQQM) are diversified vehicles, so a big weight in one
        # of them is not single-name risk — exclude them from the flag.
        single_name = positions[~positions["Ticker"].isin(BROAD_MARKET_ETFS)]
        if single_name.empty:
            broad_wt = positions["Weight"].max()
            sc3.metric("Concentration", f"{broad_wt:.1f}%", "Broad-market funds",
                       delta_color="off", help=HELP["concentration"])
        else:
            max_weight = single_name["Weight"].max()
            max_ticker = single_name.loc[single_name["Weight"].idxmax(), "Ticker"]
            if max_weight > 33:
                sc3.metric("Concentration", f"{max_weight:.1f}%", f"{max_ticker} exceeds 33%",
                           delta_color="inverse", help=HELP["concentration"])
            elif max_weight > 25:
                sc3.metric("Concentration", f"{max_weight:.1f}%", f"{max_ticker} above 25%",
                           delta_color="inverse", help=HELP["concentration"])
            else:
                sc3.metric("Concentration", f"{max_weight:.1f}%", f"Largest single name: {max_ticker}",
                           delta_color="off", help=HELP["concentration"])

        # If the portfolio's weight is dominated by broad-market funds, a high HHI
        # reflects diversified index exposure, not single-name risk — don't red-flag it.
        broad_weight = positions[positions["Ticker"].isin(BROAD_MARKET_ETFS)]["Weight"].sum()
        broad_driven = broad_weight >= 40
        if hhi and hhi > 0.25 and not broad_driven:
            hhi_label, hhi_delta = "Concentrated (>0.25)", "inverse"
        elif hhi and hhi > 0.25 and broad_driven:
            hhi_label, hhi_delta = "Broad-market weighted", "off"
        elif hhi and hhi > 0.15:
            hhi_label, hhi_delta = "Moderate", "off"
        else:
            hhi_label, hhi_delta = "Diversified", "off"
        sc4.metric("HHI Index", _fmt(hhi, ".3f"), hhi_label,
                   delta_color=hhi_delta, help=HELP["hhi"])

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Key risk metrics row ────────────────────────────────────────────────
    if not is_short_period:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Beta", f"{port_beta:.2f}", help=HELP["beta"])
        if var_data["var_1d"] is not None:
            c2.metric("1-Day VaR (95%)", f"${var_data['var_1d']:,.0f}",
                      f"{var_data['var_1d_pct']:.2f}% of portfolio", delta_color="inverse", help=HELP["var_1d"])
        else:
            c2.metric("1-Day VaR (95%)", "N/A")
        c3.metric(f"Sharpe ({period_label})", _fmt(sharpe, ".2f"),
                  help=HELP["sharpe"] + f"\n\n*Based on {lookback_days}-day lookback.*")
        c4.metric("Max Drawdown",
                  f"{max_dd * 100:.2f}%" if max_dd is not None else "N/A",
                  delta_color="inverse", help=HELP["max_dd"])
    else:
        _short_period_notice()

    st.divider()

    # ── Allocation + Sector vs S&P 500 + P&L ────────────────────────────────
    col_pie, col_sector, col_pnl = st.columns(3)

    with col_pie:
        st.subheader("Allocation")
        # Ranked horizontal bars: logo + ticker + brand-colored bar + weight %.
        alloc = positions[["Ticker", "Value"]].sort_values("Value", ascending=False).reset_index(drop=True)
        alloc_total = float(alloc["Value"].sum()) or 1.0
        max_wt = float(alloc["Value"].max()) / alloc_total * 100 or 1.0

        used_colors: list[str] = []
        rows_html: list[str] = []
        for _, prow in alloc.iterrows():
            tkr = prow["Ticker"]
            tkr_html = html.escape(str(tkr))  # defense-in-depth for unsafe_allow_html
            wt = float(prow["Value"]) / alloc_total * 100
            info = ticker_info.get(tkr) or {}
            logo_uri, color = brand_logo_and_color(tkr, info.get("website"))
            color = _distinct_color(color, used_colors)
            used_colors.append(color)
            bar_w = max(4.0, wt / max_wt * 100)   # scale to largest; keep tiny ones visible

            if logo_uri:
                icon = (f"<img src='{logo_uri}' alt='{tkr_html}' style='width:26px;height:26px;"
                        f"object-fit:contain;border-radius:5px;background:#fff;"
                        f"border:0.5px solid #eee;flex:none;'/>")
            else:
                icon = (f"<div style='width:26px;height:26px;border-radius:5px;"
                        f"background:{color};flex:none;'></div>")

            rows_html.append(
                "<div style='display:flex;align-items:center;gap:10px;padding:6px 0;"
                "border-bottom:0.5px solid #eee;'>"
                f"{icon}"
                f"<div style='width:52px;flex:none;font-weight:600;font-size:13px;color:#1a1a1a;'>{tkr_html}</div>"
                "<div style='flex:1;background:#f1f1ee;border-radius:5px;height:18px;min-width:30px;'>"
                f"<div style='width:{bar_w:.1f}%;background:{color};height:100%;border-radius:5px;'></div></div>"
                f"<div style='width:46px;flex:none;text-align:right;font-weight:600;font-size:13px;"
                f"color:#1a1a1a;font-variant-numeric:tabular-nums;'>{wt:.1f}%</div>"
                "</div>"
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)

    with col_sector:
        st.subheader("Sector vs S&P 500")
        sector_df = (
            positions.groupby("Sector")["Value"].sum().reset_index()
            .assign(**{"Portfolio %": lambda d: d["Value"] / total_value * 100})
        )
        sp500_df = pd.DataFrame.from_dict(SP500_SECTOR_WEIGHTS, orient="index", columns=["S&P 500 %"]).reset_index()
        sp500_df.columns = ["Sector", "S&P 500 %"]
        sec_merged = (
            pd.merge(sector_df[["Sector", "Portfolio %"]], sp500_df, on="Sector", how="outer")
            .fillna(0).sort_values("Portfolio %", ascending=True)
        )

        fig_sec = go.Figure()
        fig_sec.add_trace(go.Bar(
            name="Portfolio", x=sec_merged["Portfolio %"], y=sec_merged["Sector"],
            orientation="h", marker_color="#1a1a1a",
            text=sec_merged["Portfolio %"].apply(lambda v: f"{v:.1f}%"), textposition="outside",
            textfont=dict(color="#3d3d3a"),
        ))
        fig_sec.add_trace(go.Bar(
            name="S&P 500", x=sec_merged["S&P 500 %"], y=sec_merged["Sector"],
            orientation="h", marker_color="#aeb4bd",
            text=sec_merged["S&P 500 %"].apply(lambda v: f"{v:.1f}%"), textposition="outside",
            textfont=dict(color="#6b6b66"),
        ))
        fig_sec.update_layout(
            barmode="group", height=340,
            legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                        font=dict(color=INK, size=12)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=INK, margin=dict(t=40, b=30, l=0, r=20),
            xaxis=dict(title=dict(text="Weight (%)", font=dict(color=INK)),
                       tickfont=dict(color="#3d3d3a")),
            yaxis=dict(title=None, tickfont=dict(color="#3d3d3a", size=11.5)),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    with col_pnl:
        st.subheader("Unrealized P&L")
        pnl_s = positions.sort_values("PnL")
        fig_pnl = go.Figure(go.Bar(
            x=pnl_s["Ticker"], y=pnl_s["PnL"],
            marker_color=[APPLE_GREEN if v >= 0 else APPLE_RED for v in pnl_s["PnL"]],
            text=pnl_s["PnL"].apply(lambda v: f"${v:,.0f}"), textposition="outside",
        ))
        fig_pnl.add_hline(y=0, line_color=SUBTLE, line_width=1)
        _dark_chart(fig_pnl, 320)
        fig_pnl.update_layout(yaxis_title="P&L ($)", xaxis_title=None)
        st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown(DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_risk:

    if is_short_period:
        _short_period_notice()
        st.markdown(DISCLAIMER)
    else:
        # ── Down Capture alert ────────────────────────────────────────────────────
        if down_cap is not None:
            mkt_fall = 10
            port_fall = round(down_cap / 100 * mkt_fall, 1)
            if down_cap > 100:
                st.markdown(
                    f"<div class='badge badge-red'>Amplifies market downturns</div>  "
                    f"<span style='color:{APPLE_WHITE}'>When the S&P 500 falls {mkt_fall}%, this portfolio historically fell "
                    f"<b>{port_fall}%</b>.</span>",
                    unsafe_allow_html=True,
                )
            elif down_cap > 90:
                st.markdown(
                    f"<div class='badge badge-yellow'>Limited downside protection</div>  "
                    f"<span style='color:{APPLE_WHITE}'>When the S&P 500 falls {mkt_fall}%, this portfolio historically fell "
                    f"<b>{port_fall}%</b>.</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='badge badge-green'>Good downside protection</div>  "
                    f"<span style='color:{APPLE_WHITE}'>When the S&P 500 falls {mkt_fall}%, this portfolio historically fell only "
                    f"<b>{port_fall}%</b>.</span>",
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Risk metric cards ─────────────────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)

        cvar_val = cvar_data.get("cvar_1d")
        cvar_pct = cvar_data.get("cvar_1d_pct")
        if cvar_val is not None and cvar_pct is not None:
            r1.metric("CVaR (95%)", f"${cvar_val:,.0f}",
                      f"Worst-5%-day avg: {cvar_pct:.2f}%",
                      delta_color="inverse", help=HELP["cvar"])
        else:
            r1.metric("CVaR (95%)", "N/A")

        r2.metric("Calmar Ratio",  _fmt(calmar,  ".2f"), help=HELP["calmar"])
        r3.metric("Treynor Ratio", _fmt(treynor, ".2f"), help=HELP["treynor"])
        r4.metric(f"Sharpe ({period_label})", _fmt(sharpe, ".2f"),
                  help=HELP["sharpe"] + f"\n\n*Based on {lookback_days}-day lookback.*")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Ulcer Index", _fmt(ulcer, ".2f"),
                  "Low (<5)" if (ulcer and ulcer < 5) else
                  ("Moderate (5–15)" if (ulcer and ulcer < 15) else "High (>15)"),
                  delta_color="inverse", help=HELP["ulcer"])
        r6.metric("Pain Ratio",  _fmt(pain,  ".2f"), help=HELP["pain_ratio"])
        r7.metric("Omega Ratio", _fmt(omega, ".2f"), help=HELP["omega"])
        r8.metric("Annualised Volatility",
                  f"{ann_vol * 100:.2f}%" if ann_vol else "N/A", help=HELP["ann_vol"])

        st.divider()

        # ── Beta gauge + VaR multi-confidence ────────────────────────────────────
        col_gauge, col_var = st.columns(2)

        with col_gauge:
            st.subheader("Portfolio Beta")
            gauge_max = max(3.0, round(port_beta * 1.6, 1))
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=port_beta,
                delta={"reference": 1.0, "suffix": " vs market", "font": {"size": 14}},
                number={"font": {"size": 34, "color": APPLE_WHITE}},
                gauge={
                    "axis": {"range": [0, gauge_max], "tickcolor": APPLE_GRAY},
                    "bar":  {"color": APPLE_WHITE, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 0.5],      "color": "#d6ecdd"},
                        {"range": [0.5, 1.0],    "color": "#e9f5ed"},
                        {"range": [1.0, 1.5],    "color": "#fcf3d9"},
                        {"range": [1.5, gauge_max], "color": "#fbe9e9"},
                    ],
                    "threshold": {"line": {"color": APPLE_WHITE, "width": 3},
                                  "thickness": 0.75, "value": 1.0},
                },
            ))
            fig_gauge.update_layout(
                height=280, paper_bgcolor="rgba(0,0,0,0)", font_color=APPLE_GRAY,
                margin=dict(t=20, b=52, l=30, r=30),
                annotations=[
                    dict(x=0.1, y=-0.12, text="Defensive", showarrow=False, font=dict(size=9, color=APPLE_GRAY)),
                    dict(x=0.38, y=-0.12, text="Conservative", showarrow=False, font=dict(size=9, color=APPLE_GRAY)),
                    dict(x=0.65, y=-0.12, text="Aggressive", showarrow=False, font=dict(size=9, color=APPLE_GRAY)),
                    dict(x=0.90, y=-0.12, text="Speculative", showarrow=False, font=dict(size=9, color=APPLE_GRAY)),
                ],
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption("Beta = 1.0 means market-level risk — not low risk. Values > 1.5 amplify both gains and losses.")

        with col_var:
            st.subheader("VaR & CVaR — Confidence Levels")
            if var_multi:
                var_df = pd.DataFrame([
                    {"Level": "90%", "VaR ($)": var_multi.get("var_90"), "CVaR ($)": var_multi.get("cvar_90")},
                    {"Level": "95%", "VaR ($)": var_multi.get("var_95"), "CVaR ($)": var_multi.get("cvar_95")},
                    {"Level": "99%", "VaR ($)": var_multi.get("var_99"), "CVaR ($)": var_multi.get("cvar_99")},
                ]).dropna()
                fig_vm = go.Figure()
                fig_vm.add_trace(go.Bar(name="VaR",  x=var_df["Level"], y=var_df["VaR ($)"],
                                        marker_color=APPLE_WHITE,
                                        text=var_df["VaR ($)"].apply(lambda v: f"${v:,.0f}"),
                                        textposition="outside"))
                fig_vm.add_trace(go.Bar(name="CVaR", x=var_df["Level"], y=var_df["CVaR ($)"],
                                        marker_color=APPLE_RED,
                                        text=var_df["CVaR ($)"].apply(lambda v: f"${v:,.0f}"),
                                        textposition="outside"))
                fig_vm.update_layout(barmode="group", yaxis_title="Expected Loss ($)",
                                     showlegend=True, legend=dict(orientation="h", y=1.1))
                _dark_chart(fig_vm, 280)
                st.plotly_chart(fig_vm, use_container_width=True)

        st.divider()

        # ── Tail Risk histogram ───────────────────────────────────────────────────
        st.subheader("Daily Return Distribution")
        from calculations import _portfolio_daily_returns as _pdr
        port_ret_series = _pdr(pdf, positions)
        if port_ret_series is not None and len(port_ret_series) >= 30:
            var_thresh = -var_data["var_1d_pct"] / 100 if var_data["var_1d_pct"] else None
            cvar_thresh = -cvar_data["cvar_1d_pct"] / 100 if cvar_data["cvar_1d_pct"] else None
            ret_vals = port_ret_series.dropna().values * 100
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=ret_vals, nbinsx=60, name="Daily Returns",
                                            marker_color="#bdbdb7", opacity=0.85))
            if var_thresh is not None:
                tail_vals = ret_vals[ret_vals <= var_thresh * 100]
                fig_hist.add_trace(go.Histogram(x=tail_vals, nbinsx=20, name="Worst 5%",
                                                marker_color=APPLE_RED, opacity=0.9))
                fig_hist.add_vline(x=var_thresh * 100, line_dash="dash", line_color=APPLE_WHITE,
                                   annotation_text=f"VaR 95%: {var_thresh * 100:.2f}%",
                                   annotation_position="top right")
            if cvar_thresh is not None:
                fig_hist.add_vline(x=cvar_thresh * 100, line_dash="dot", line_color=APPLE_RED,
                                   annotation_text=f"CVaR 95%: {cvar_thresh * 100:.2f}%",
                                   annotation_position="bottom left")
            fig_hist.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                                   barmode="overlay", showlegend=True,
                                   legend=dict(orientation="h", y=1.18))
            _dark_chart(fig_hist, 280)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Stress tests ──────────────────────────────────────────────────────────
        st.subheader("Stress Test Scenarios")
        if not stress_df.empty:
            for _, srow in stress_df.iterrows():
                pct = float(srow["Est. Portfolio Loss (%)"])
                val = float(srow["Est. Portfolio Loss ($)"])
                if pct < -30:
                    badge_cls = "badge badge-red"
                    badge_text = "SEVERE"
                elif pct < -15:
                    badge_cls = "badge badge-yellow"
                    badge_text = "ELEVATED"
                else:
                    badge_cls = "badge badge-gray"
                    badge_text = "MODERATE"
                st.markdown(
                    f"<div style='margin: 8px 0'>"
                    f"<span class='{badge_cls}'>{badge_text}</span>  "
                    f"<b style='color:{APPLE_WHITE}'>{srow['Scenario']}</b>  "
                    f"<span style='color:{APPLE_GRAY}'>(Market: {srow['Market Shock']})</span> — "
                    f"Est. loss: <b style='color:{APPLE_RED if pct < 0 else APPLE_WHITE}'>${val:,.0f}</b> "
                    f"<span style='color:{APPLE_GRAY}'>({pct:.1f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Rolling charts ────────────────────────────────────────────────────────
        if not rolling.empty:
            col_rs, col_rv = st.columns(2)
            with col_rs:
                st.subheader("Rolling Sharpe (63d)")
                fig_rs = go.Figure(go.Scatter(
                    x=rolling.index, y=rolling["Rolling Sharpe"],
                    line=dict(color=APPLE_WHITE, width=1.8), fill="tozeroy",
                    fillcolor="rgba(0,0,0,0.06)",
                    hovertemplate="%{y:.2f}<extra></extra>",
                ))
                fig_rs.add_hline(y=1.0, line_dash="dash", line_color=APPLE_GREEN,
                                 annotation_text="1.0", annotation_position="top right")
                fig_rs.add_hline(y=0.0, line_color=SUBTLE, line_width=0.5)
                _dark_chart(fig_rs)
                fig_rs.update_layout(yaxis_title="Sharpe")
                st.plotly_chart(fig_rs, use_container_width=True)

            with col_rv:
                st.subheader("Rolling Volatility (63d, ann.)")
                fig_rv = go.Figure(go.Scatter(
                    x=rolling.index, y=rolling["Rolling Volatility (%)"],
                    line=dict(color=APPLE_RED, width=1.8), fill="tozeroy",
                    fillcolor="rgba(255,69,58,0.10)",
                    hovertemplate="%{y:.1f}%<extra></extra>",
                ))
                _dark_chart(fig_rv)
                fig_rv.update_layout(yaxis_title="Volatility (%)")
                st.plotly_chart(fig_rv, use_container_width=True)

        st.divider()

        # ── Drawdown chart ────────────────────────────────────────────────────────
        st.subheader("Drawdown from Rolling Peak")
        if dd_series is not None and not dd_series.empty:
            fig_dd = go.Figure(go.Scatter(
                x=dd_series.index, y=dd_series * 100,
                fill="tozeroy", fillcolor="rgba(255,69,58,0.18)",
                line=dict(color=APPLE_RED, width=1.5),
                hovertemplate="%{y:.2f}%<extra></extra>",
            ))
            fig_dd.add_hline(y=0, line_color=SUBTLE, line_width=0.5)
            _dark_chart(fig_dd, 220)
            fig_dd.update_layout(yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

        # ── Correlation heatmap ───────────────────────────────────────────────────
        if not corr.empty and corr.shape[0] > 1:
            st.subheader("Return Correlation Matrix")
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdYlGn_r", zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text:.2f}",
                colorbar=dict(title="ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
            ))
            fig_corr.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                                   font_color=INK, xaxis=dict(tickfont=dict(color=INK, size=12)), yaxis=dict(tickfont=dict(color=INK, size=12)), margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("What do these metrics mean?"):
            st.markdown(SECTION_EXPLAINERS["risk"])

        st.markdown(DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · BENCHMARKS (active management focus)
# ══════════════════════════════════════════════════════════════════════════════

with tab_bench:

    if is_short_period:
        _short_period_notice()
        st.markdown(DISCLAIMER)
    else:
        alpha = bench_stats.get("alpha")
        r2    = bench_stats.get("r2")
        te    = bench_stats.get("tracking_error")

        b1, b2, b3 = st.columns(3)
        b1.metric("Jensen's Alpha",  f"{alpha * 100:+.2f}%" if alpha is not None else "N/A",
                  None if alpha is None else ("Above CAPM" if alpha > 0 else "Below CAPM"),
                  delta_color="normal" if (alpha is not None and alpha > 0) else "inverse", help=HELP["alpha"])
        b2.metric("R² vs S&P 500",  f"{r2 * 100:.1f}%" if r2 is not None else "N/A", help=HELP["r2"])
        b3.metric("Tracking Error", f"{te * 100:.2f}%" if te is not None else "N/A", help=HELP["tracking_error"])

        b4, b5, b6 = st.columns(3)
        b4.metric("Up Capture", f"{up_cap:.1f}%" if up_cap is not None else "N/A",
                  "Captures upside" if (up_cap and up_cap > 100) else "Lags on rallies",
                  delta_color="normal" if (up_cap and up_cap > 100) else "inverse", help=HELP["up_capture"])
        if down_cap is not None:
            if down_cap > 100:   dc_delta, dc_color = "Amplifies losses", "inverse"
            elif down_cap > 90:  dc_delta, dc_color = "Caution",           "off"
            else:                dc_delta, dc_color = "Good protection",   "normal"
            b5.metric("Down Capture", f"{down_cap:.1f}%", dc_delta, delta_color=dc_color, help=HELP["down_capture"])
        else:
            b5.metric("Down Capture", "N/A")
        b6.metric("Info Ratio", _fmt(info_ratio, ".2f"),
                  "Skilled active" if (info_ratio and info_ratio > 0.5) else
                  ("Marginal" if (info_ratio and info_ratio > 0) else "Lags benchmark"),
                  delta_color="normal" if (info_ratio and info_ratio > 0.5) else "inverse", help=HELP["info_ratio"])

        with st.expander("What do these metrics mean?"):
            st.markdown(SECTION_EXPLAINERS["benchmarks"])

        st.divider()

        # ── Rolling Beta & Correlation ────────────────────────────────────────────
        if len(roll_beta_series) > 0 or len(roll_corr_series) > 0:
            rb_col, rc_col = st.columns(2)

            with rb_col:
                st.subheader("Rolling Beta vs S&P 500 (90d)")
                if len(roll_beta_series) > 0:
                    fig_rb = go.Figure(go.Scatter(
                        x=roll_beta_series.index, y=roll_beta_series.values,
                        line=dict(color=APPLE_WHITE, width=1.8),
                        hovertemplate="%{y:.2f}<extra></extra>",
                    ))
                    fig_rb.add_hline(y=1.0, line_dash="dash", line_color=APPLE_GRAY,
                                     annotation_text="Market β=1.0")
                    _dark_chart(fig_rb)
                    fig_rb.update_layout(yaxis_title="Rolling Beta", xaxis_title="Date")
                    st.plotly_chart(fig_rb, use_container_width=True)

            with rc_col:
                st.subheader("Rolling Correlation vs S&P 500 (90d)")
                if len(roll_corr_series) > 0:
                    fig_rc = go.Figure(go.Scatter(
                        x=roll_corr_series.index, y=roll_corr_series.values,
                        line=dict(color=APPLE_BLUE, width=1.8),
                        fill="tozeroy", fillcolor="rgba(0,0,0,0.06)",
                        hovertemplate="%{y:.2f}<extra></extra>",
                    ))
                    fig_rc.add_hline(y=0, line_color=SUBTLE, line_width=0.5)
                    _dark_chart(fig_rc)
                    fig_rc.update_layout(yaxis_title="Correlation (ρ)", xaxis_title="Date",
                                         yaxis_range=[-1.1, 1.1])
                    st.plotly_chart(fig_rc, use_container_width=True)

        # ── Active Return bar chart ───────────────────────────────────────────────
        st.subheader("Monthly Active Return vs S&P 500")
        if port_cum is not None and not port_cum.empty and PRIMARY_BENCH in bench_df.columns:
            from calculations import _portfolio_daily_returns as _pdr
            pr = _pdr(pdf, positions)
            if pr is not None:
                bench_ret = bench_df[PRIMARY_BENCH].pct_change().dropna()
                aligned = pd.DataFrame({"port": pr, "bench": bench_ret}).dropna()
                aligned.index = pd.DatetimeIndex(aligned.index)
                monthly = aligned.resample("ME").apply(lambda x: (1 + x).prod() - 1)
                monthly["active"] = monthly["port"] - monthly["bench"]
                fig_ar = go.Figure(go.Bar(
                    x=monthly.index, y=monthly["active"] * 100,
                    marker_color=[APPLE_GREEN if v >= 0 else APPLE_RED for v in monthly["active"]],
                    hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra></extra>",
                ))
                fig_ar.add_hline(y=0, line_color=SUBTLE, line_width=0.5)
                _dark_chart(fig_ar, 280)
                fig_ar.update_layout(yaxis_title="Active Return (%)", xaxis_title="Date")
                st.plotly_chart(fig_ar, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────────
        if alpha is not None:
            st.markdown("#### Relative Performance Summary (vs S&P 500)")
            summary_data = [
                {"Metric": "Jensen's Alpha", "Value": f"{alpha * 100:+.2f}%",
                 "Interpretation": "Above market expectation" if alpha > 0 else "Below market expectation"},
                {"Metric": "R-Squared", "Value": f"{r2 * 100:.1f}%",
                 "Interpretation": "Highly index-like" if r2 > 0.9 else ("Moderately correlated" if r2 > 0.6 else "Low benchmark correlation")},
                {"Metric": "Tracking Error", "Value": f"{te * 100:.2f}%",
                 "Interpretation": "Tight index tracking" if te < 0.03 else ("Active range" if te < 0.10 else "Highly active")},
                {"Metric": "Up Capture", "Value": f"{up_cap:.1f}%" if up_cap else "N/A",
                 "Interpretation": "Captures upside well" if (up_cap and up_cap > 100) else "Underperforms in rallies"},
                {"Metric": "Down Capture", "Value": f"{down_cap:.1f}%" if down_cap else "N/A",
                 "Interpretation": "Good downside protection" if (down_cap and down_cap < 90) else
                 ("Limited protection" if (down_cap and down_cap < 100) else "Amplifies market losses")},
                {"Metric": "Information Ratio", "Value": _fmt(info_ratio, ".2f"),
                 "Interpretation": "Skilled active mgmt" if (info_ratio and info_ratio > 0.5) else
                 ("Marginal active value" if (info_ratio and info_ratio > 0) else "Benchmark dominates")},
            ]
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.markdown(DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_mc:

    st.subheader("Monte Carlo Portfolio Projection")

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    mc_years   = mc_col1.slider("Time Horizon (Years)", 5, 40, 20, 5)
    mc_contrib = mc_col2.number_input(
        "Monthly Contribution ($)", min_value=0, max_value=50_000, value=0, step=100,
        help="Additional cash added each month (every 21 trading days).",
    )
    mc_col3.metric("Starting Value", f"${total_value:,.2f}")

    with st.expander("Goal Planning Mode"):
        gc1, gc2 = st.columns(2)
        goal_value = gc1.number_input("Target Portfolio Value ($)", min_value=1000,
                                       value=int(total_value * 3), step=10000)
        goal_return_pct = mc_expected_return_pct

        def _required_contribution(pv, fv, r_annual, n_years):
            r_monthly = (1 + r_annual) ** (1 / 12) - 1
            n_months = n_years * 12
            if r_monthly == 0:
                return max(0, (fv - pv) / n_months)
            fv_pv = pv * (1 + r_monthly) ** n_months
            if fv <= fv_pv:
                return 0.0
            return (fv - fv_pv) * r_monthly / ((1 + r_monthly) ** n_months - 1)

        for label, rate_adj in [("Median (9%)", 0), ("Conservative (6%)", -0.03), ("Pessimistic (4%)", -0.05)]:
            adj_rate = max(0.01, goal_return_pct / 100 + rate_adj)
            req = _required_contribution(total_value, goal_value, adj_rate, mc_years)
            gc2.markdown(f"**{label}:** ${req:,.0f}/month to reach ${goal_value:,.0f} in {mc_years}y")

    sor_toggle = st.checkbox(
        "Show Sequence-of-Returns Risk (worst years first)",
        help="Reverses historical return order to simulate worst-case timing.",
    )

    mc_expected_return = None if mc_use_hist else mc_expected_return_pct / 100
    mc_inflation_rate  = mc_inflation_rate_pct / 100

    if is_short_period:
        st.warning(f"Monte Carlo requires daily-frequency data — switch to **3M** or longer.")
    else:
        mc_result = cached_monte_carlo(
            pdf, positions, total_value, mc_years, mc_sims,
            float(mc_contrib), mc_expected_return, mc_inflation_rate,
        )

        if mc_result is not None:
            mc_df      = mc_result["df"]
            mc_df_real = mc_result["df_real"]
            hist_ann   = mc_result["historical_ann_return"]
            used_ann   = mc_result["used_ann_return"]
            was_capped = mc_result["capped"]

            if was_capped:
                st.markdown(
                    f"<div class='badge badge-yellow'>Return cap applied</div>  "
                    f"<span style='color:{APPLE_WHITE}'>Recent returns are unusually high "
                    f"({hist_ann * 100:.1f}% annualised). Projection capped at "
                    f"<b>{used_ann * 100:.1f}%</b>.</span>",
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Expected return: **{used_ann * 100:.1f}%** · "
                f"Historical vol: **{ann_vol * 100:.1f}%** · "
                f"Inflation: **{mc_inflation_rate * 100:.1f}%**"
            )

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p95"], mode="lines",
                                        line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p5"], mode="lines",
                                        fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
                                        line=dict(color="rgba(0,0,0,0)"), name="5th–95th pct"))
            fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p75"], mode="lines",
                                        line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p25"], mode="lines",
                                        fill="tonexty", fillcolor="rgba(42,120,214,0.28)",
                                        line=dict(color="rgba(0,0,0,0)"), name="25th–75th pct"))
            fig_mc.add_trace(go.Scatter(x=mc_df["year"], y=mc_df["p50"], mode="lines",
                                        line=dict(color="#2a78d6", width=2.6),
                                        name="Median (nominal)"))
            fig_mc.add_trace(go.Scatter(x=mc_df_real["year"], y=mc_df_real["p50"], mode="lines",
                                        line=dict(color="#b8860b", width=1.8, dash="dash"),
                                        name=f"Median (real, {mc_inflation_rate * 100:.1f}% infl.)"))
            fig_mc.add_hline(y=total_value, line_dash="dot", line_color=SUBTLE,
                             annotation_text=f"Start ${total_value:,.0f}",
                             annotation_position="top right")
            _dark_chart(fig_mc, 460)
            fig_mc.update_layout(
                yaxis_title="Portfolio Value ($)", xaxis_title="Years from Today",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=50, b=20, l=10, r=10),
            )
            fig_mc.update_yaxes(tickprefix="$", tickformat=",.0f")

            if sor_toggle:
                from calculations import _portfolio_daily_returns as _pdr
                pr = _pdr(pdf, positions)
                if pr is not None and len(pr) >= 30:
                    pr_reversed = pr.iloc[::-1].values
                    daily_std_sor = float(pr.std())
                    hist_daily_mean_sor = float(pr.mean())
                    n_days_sor = mc_years * 252
                    rng_sor = np.random.default_rng(99)
                    block = np.tile(pr_reversed, n_days_sor // len(pr_reversed) + 1)[:n_days_sor]
                    rand_sor = rng_sor.normal(hist_daily_mean_sor, daily_std_sor, (n_days_sor, mc_sims))
                    rand_sor[:len(block)] = block[:, np.newaxis]
                    paths_sor = np.empty((n_days_sor + 1, mc_sims))
                    paths_sor[0] = total_value
                    for t in range(1, n_days_sor + 1):
                        paths_sor[t] = paths_sor[t - 1] * (1 + rand_sor[t - 1])
                        if float(mc_contrib) > 0 and t % 21 == 0:
                            paths_sor[t] += float(mc_contrib)
                    sor_year = np.linspace(0, mc_years, n_days_sor + 1)
                    sor_median = np.percentile(paths_sor, 50, axis=1)
                    fig_mc.add_trace(go.Scatter(
                        x=sor_year, y=sor_median, mode="lines",
                        line=dict(color=APPLE_RED, width=1.8, dash="longdash"),
                        name="Median (worst years first)",
                    ))

            st.plotly_chart(fig_mc, use_container_width=True)

            st.markdown("#### Projected Outcomes by Horizon")
            horizon_rows = []
            for y in [1, 3, 5, 10, 20, 30]:
                if y > mc_years:
                    continue
                idx = min(int(y * 252), len(mc_df) - 1)
                row_n = mc_df.iloc[idx]
                row_r = mc_df_real.iloc[idx]
                horizon_rows.append({
                    "Horizon":             f"{y}yr",
                    "Bear (5th)":          f"${row_n['p5']:,.0f}",
                    "Conservative (25th)": f"${row_n['p25']:,.0f}",
                    "Median (50th)":       f"${row_n['p50']:,.0f}",
                    "Optimistic (75th)":   f"${row_n['p75']:,.0f}",
                    "Bull (95th)":         f"${row_n['p95']:,.0f}",
                    "Median (real $)":     f"${row_r['p50']:,.0f}",
                })
            st.dataframe(pd.DataFrame(horizon_rows), use_container_width=True, hide_index=True)
            st.caption(f"Real values deflated at {mc_inflation_rate * 100:.1f}% annual inflation.")

    with st.expander("How to read this simulation"):
        st.markdown(SECTION_EXPLAINERS["monte_carlo"])

    st.markdown(DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════

with tab_holdings:

    st.subheader("Position Details")

    display_positions = positions.copy()
    display_positions["VaR 1D (95%)"]    = display_positions["Ticker"].map(lambda t: pos_var.get(t))
    display_positions["Risk Contrib %"]  = display_positions["Ticker"].map(lambda t: mrc.get(t))

    display_cols = [
        "Ticker", "Sector", "Shares", "Avg_Cost", "Current_Price",
        "Value", "Cost_Basis", "PnL", "PnL_Pct", "Weight", "Beta",
        "Dividend_Yield", "Week52High", "Week52Low",
        "VaR 1D (95%)", "Risk Contrib %",
    ]
    avail_cols = [c for c in display_cols if c in display_positions.columns]
    display_df = display_positions[avail_cols].rename(columns={
        "Avg_Cost":       "Avg Cost",
        "Current_Price":  "Price",
        "Cost_Basis":     "Cost Basis",
        "PnL":            "P&L ($)",
        "PnL_Pct":        "P&L (%)",
        "Weight":         "Weight (%)",
        "Dividend_Yield": "Div Yield",
        "Week52High":     "52W High",
        "Week52Low":      "52W Low",
    })

    fmt_map = {
        "Avg Cost":       "${:.2f}",
        "Price":          "${:.2f}",
        "Value":          "${:,.2f}",
        "Cost Basis":     "${:,.2f}",
        "P&L ($)":        "${:,.2f}",
        "P&L (%)":        "{:.2f}%",
        "Weight (%)":     "{:.2f}%",
        "Beta":           "{:.2f}",
        "Shares":         "{:.4f}",
        "Div Yield":      "{:.2%}",
        "52W High":       "${:.2f}",
        "52W Low":        "${:.2f}",
        "VaR 1D (95%)":   "${:,.0f}",
        "Risk Contrib %": "{:.1f}%",
    }
    active_fmt = {k: v for k, v in fmt_map.items() if k in display_df.columns}

    display_df = display_df.replace({None: np.nan})

    def _pnl_color(val):
        if not isinstance(val, (int, float, np.floating)):
            return ""
        return f"color: {APPLE_GREEN}" if val > 0 else (f"color: {APPLE_RED}" if val < 0 else "")

    styled = (
        display_df.style
        .format(active_fmt, na_rep="—")
        .map(_pnl_color, subset=["P&L ($)", "P&L (%)"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Concentration flags
    for _, row in positions.iterrows():
        if row["Weight"] > 33:
            st.markdown(
                f"<div style='margin: 6px 0'><span class='badge badge-red'>Over-Concentrated</span>  "
                f"<b style='color:{APPLE_WHITE}'>{html.escape(str(row['Ticker']))}</b> is {row['Weight']:.1f}% — exceeds 33%.</div>",
                unsafe_allow_html=True,
            )
        elif row["Weight"] > 25:
            st.markdown(
                f"<div style='margin: 6px 0'><span class='badge badge-yellow'>Caution</span>  "
                f"<b style='color:{APPLE_WHITE}'>{html.escape(str(row['Ticker']))}</b> is {row['Weight']:.1f}% — exceeds 25%.</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── 52-Week range progress ────────────────────────────────────────────────
    if "Week52High" in positions.columns and positions["Week52High"].notna().any():
        st.subheader("52-Week Price Range")
        for _, row in positions.iterrows():
            lo, hi, price = row.get("Week52Low"), row.get("Week52High"), row["Current_Price"]
            if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
                continue
            rng = hi - lo
            if rng <= 0:
                continue
            pos_pct = (price - lo) / rng * 100
            if pos_pct < 30:
                badge_cls, badge_lbl = "badge badge-red",    "Near Low"
            elif pos_pct < 70:
                badge_cls, badge_lbl = "badge badge-yellow", "Mid-Range"
            else:
                badge_cls, badge_lbl = "badge badge-green",  "Near High"
            st.markdown(
                f"<b style='color:{APPLE_WHITE}'>{html.escape(str(row['Ticker']))}</b>  "
                f"<span style='color:{APPLE_GRAY}'>${price:.2f}  ·  "
                f"Low ${lo:.2f}  ·  High ${hi:.2f}</span>  "
                f"<span class='{badge_cls}'>{badge_lbl}</span>",
                unsafe_allow_html=True,
            )
            st.progress(int(min(100, max(0, pos_pct))))

    st.divider()

    # ── Marginal Risk Contribution chart ──────────────────────────────────────
    if mrc:
        st.subheader("Risk Contribution by Position")
        mrc_df = pd.DataFrame.from_dict(mrc, orient="index", columns=["Risk %"]).reset_index()
        mrc_df.columns = ["Ticker", "Risk %"]
        mrc_df = mrc_df.sort_values("Risk %", ascending=False)
        fig_mrc = go.Figure(go.Bar(
            x=mrc_df["Ticker"], y=mrc_df["Risk %"],
            marker_color=[APPLE_RED if v > 30 else (YELLOW if v > 20 else APPLE_GREEN) for v in mrc_df["Risk %"]],
            text=mrc_df["Risk %"].apply(lambda v: f"{v:.1f}%"), textposition="outside",
        ))
        fig_mrc.add_hline(y=30, line_dash="dash", line_color=APPLE_RED,
                          annotation_text="30% threshold", annotation_position="top right")
        _dark_chart(fig_mrc, 280)
        fig_mrc.update_layout(yaxis_title="% of Total Portfolio Risk", xaxis_title=None)
        st.plotly_chart(fig_mrc, use_container_width=True)
        st.caption("Positions above 30% of total risk warrant review.")

    st.divider()

    # ── Beta Contribution chart ───────────────────────────────────────────────
    st.subheader("Beta Contribution by Position")
    beta_df = positions[["Ticker", "Beta", "Weight", "Beta_Contribution"]].sort_values(
        "Beta_Contribution", ascending=False)
    fig_beta = go.Figure(go.Bar(
        x=beta_df["Ticker"], y=beta_df["Beta_Contribution"],
        marker=dict(color=beta_df["Beta"].astype(float), colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Raw β", tickfont=dict(color=APPLE_GRAY))),
        text=beta_df["Beta_Contribution"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—"), textposition="outside",
    ))
    fig_beta.add_hline(y=port_beta, line_dash="dash", line_color=APPLE_GRAY,
                       annotation_text=f"Portfolio β = {port_beta:.2f}", annotation_position="top right")
    _dark_chart(fig_beta, 300)
    fig_beta.update_layout(yaxis_title="Beta Contribution", xaxis_title=None)
    st.plotly_chart(fig_beta, use_container_width=True)

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("Export Risk Report")

    def build_report() -> bytes:
        buf = io.StringIO()
        buf.write("PORTFOLIO RISK REPORT\n")
        buf.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        buf.write(f"Lookback Period: {period_label}\n\n")
        summary = {
            "Total Portfolio Value":    f"${total_value:,.2f}",
            "Total Unrealized P&L":     f"${total_pnl:,.2f}",
            "Total P&L %":              f"{total_pnl_pct:.2f}%",
            "Risk Score":               f"{risk_score_val}/10" if risk_score_val else "N/A",
            "Diversification Score":    f"{div_score}/10" if div_score else "N/A",
            "Portfolio Beta":           f"{port_beta:.2f}",
            "Sharpe Ratio":             _fmt(sharpe,  ".2f"),
            "Sortino Ratio":            _fmt(sortino, ".2f"),
            "Calmar Ratio":             _fmt(calmar,  ".2f"),
            "Treynor Ratio":            _fmt(treynor, ".2f"),
            "Ulcer Index":              _fmt(ulcer,   ".2f"),
            "Pain Ratio":               _fmt(pain,    ".2f"),
            "Omega Ratio":              _fmt(omega,   ".2f"),
            "Max Drawdown":             f"{max_dd * 100:.2f}%" if max_dd is not None else "N/A",
            "Annualised Volatility":    f"{ann_vol * 100:.2f}%" if ann_vol else "N/A",
            "HHI Concentration":        f"{hhi:.3f}",
            "1-Day VaR (95%)":          f"${var_data['var_1d']:,.2f}" if var_data["var_1d"] else "N/A",
            "1-Day CVaR (95%)":         f"${cvar_data['cvar_1d']:,.2f}" if cvar_data["cvar_1d"] else "N/A",
            "Jensen's Alpha":           f"{bench_stats['alpha'] * 100:+.2f}%" if bench_stats["alpha"] else "N/A",
            "R² vs S&P 500":            f"{bench_stats['r2'] * 100:.1f}%" if bench_stats["r2"] else "N/A",
            "Tracking Error":           f"{bench_stats['tracking_error'] * 100:.2f}%" if bench_stats["tracking_error"] else "N/A",
            "Up Capture Ratio":         f"{up_cap:.1f}%" if up_cap else "N/A",
            "Down Capture Ratio":       f"{down_cap:.1f}%" if down_cap else "N/A",
            "Information Ratio":        _fmt(info_ratio, ".2f"),
            "Risk-Free Rate Used":      f"{risk_free_rate * 100:.2f}%",
        }
        def _xl_safe(df: pd.DataFrame) -> pd.DataFrame:
            # Excel formula-injection guard for text cells starting with =, +, @
            df = df.copy()
            for c in df.columns[df.dtypes == object]:
                df[c] = df[c].map(
                    lambda v: "'" + v if isinstance(v, str) and v[:1] in "=+@" else v)
            return df

        pd.DataFrame.from_dict(summary, orient="index", columns=["Value"]).to_csv(buf)
        buf.write("\nPOSITION DETAILS\n")
        _xl_safe(positions).to_csv(buf, index=False)
        buf.write("\nCORRELATION MATRIX\n")
        if not corr.empty:
            corr.round(4).to_csv(buf)
        if not stress_df.empty:
            buf.write("\nSTRESS TEST RESULTS\n")
            _xl_safe(stress_df).to_csv(buf, index=False)
        return buf.getvalue().encode()

    st.download_button(
        "Download Full Risk Report (CSV)",
        data=build_report(),
        file_name=f"portfolio_risk_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    with st.expander("About position-level metrics"):
        st.markdown(SECTION_EXPLAINERS["holdings"])

    st.markdown(DISCLAIMER)
