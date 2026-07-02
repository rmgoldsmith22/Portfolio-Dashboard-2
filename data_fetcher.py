from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

# NOTE: yfinance >= 0.2.54 manages its own (curl_cffi) session; passing a custom
# requests.Session is deprecated there and can break silently. Let yfinance handle it.


def _close_from_hist(hist: pd.DataFrame) -> pd.Series | None:
    col = next((c for c in ["Close", "close", "Adj Close"] if c in hist.columns), None)
    return hist[col].copy() if col else None


def fetch_price_history(
    tickers: list,
    period: str = "1y",
    interval: str = "1d",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch closing prices at the requested period + interval.
    One batched yf.download() for all tickers, per-ticker fallback for misses.
    Returns (price_df, failed_tickers). No Streamlit calls.
    """
    tickers = list(dict.fromkeys(tickers))
    price_data: dict[str, pd.Series] = {}
    failed: list[str] = []

    # Fast path: one batched download for every ticker at once.
    try:
        raw = yf.download(
            tickers, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=True,
        )
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:  # single ticker → flat columns
                closes = raw[["Close"]].rename(columns={"Close": tickers[0]})
            for t in tickers:
                if t in closes.columns:
                    series = closes[t].dropna()
                    if not series.empty:
                        series.index = pd.DatetimeIndex(series.index.values)
                        price_data[t] = series.rename(t)
    except Exception:
        pass  # fall through to per-ticker fallback

    # Fallback: per-ticker fetch for anything the batch missed.
    for ticker in tickers:
        if ticker in price_data:
            continue
        close = None
        try:
            hist = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=True,
            )
            if not hist.empty:
                close = _close_from_hist(hist)
        except Exception:
            pass
        if close is None or close.empty:
            failed.append(ticker)
            continue
        close.index = pd.DatetimeIndex(close.index.values)
        price_data[ticker] = close.rename(ticker)
        time.sleep(0.1)  # gentle pacing only on the slow path

    if not price_data:
        return pd.DataFrame(), failed

    df = pd.DataFrame(price_data).dropna(how="all")
    df.sort_index(inplace=True)
    return df, failed


def fetch_ticker_info(ticker: str) -> dict:
    """Returns current_price, market_cap, beta, sector, dividend_yield, 52-week range. All may be None."""
    empty = {
        "current_price": None, "market_cap": None, "beta": None,
        "sector": None, "dividend_yield": None,
        "week_52_high": None, "week_52_low": None,
        "website": None, "long_name": None,
    }
    try:
        info = yf.Ticker(ticker).info
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice")
            or info.get("previousClose")
        )
        beta = info.get("beta") or info.get("beta3Year")
        # yfinance >= 0.2.54 returns `dividendYield` in PERCENT (0.52 == 0.52%) while
        # `trailingAnnualDividendYield` stays a fraction. Normalize both to a fraction.
        div_yield = info.get("dividendYield")
        if div_yield is not None:
            div_yield = float(div_yield) / 100.0
        else:
            div_yield = info.get("trailingAnnualDividendYield")
        return {
            "current_price":  float(price) if price is not None else None,
            "market_cap":     info.get("marketCap"),
            "beta":           float(beta) if beta is not None else None,
            "sector":         info.get("sector") or info.get("categoryName"),
            "dividend_yield": float(div_yield) if div_yield is not None else None,
            "week_52_high":   info.get("fiftyTwoWeekHigh"),
            "week_52_low":    info.get("fiftyTwoWeekLow"),
            "website":        info.get("website"),
            "long_name":      info.get("longName") or info.get("shortName"),
        }
    except Exception:
        return empty
