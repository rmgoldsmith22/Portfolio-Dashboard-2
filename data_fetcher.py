from __future__ import annotations

import time

import pandas as pd
import requests
import yfinance as yf

# Session with browser-like headers reduces Yahoo Finance rate-limiting on cloud hosts
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


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
    Tries Ticker.history() then yf.download() as fallback.
    Returns (price_df, failed_tickers). No Streamlit calls.
    """
    price_data: dict[str, pd.Series] = {}
    failed: list[str] = []

    for ticker in tickers:
        close = None

        # Attempt 1: Ticker.history()
        try:
            hist = yf.Ticker(ticker, session=_SESSION).history(
                period=period, interval=interval, auto_adjust=True,
            )
            if not hist.empty:
                close = _close_from_hist(hist)
        except Exception:
            pass

        # Attempt 2: yf.download() fallback
        if close is None or close.empty:
            try:
                raw = yf.download(
                    ticker, period=period, interval=interval, auto_adjust=True,
                    progress=False, threads=False,
                )
                if not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.droplevel(1)
                    close = _close_from_hist(raw)
            except Exception:
                pass

        if close is None or (hasattr(close, "empty") and close.empty):
            failed.append(ticker)
            continue

        close.index = pd.DatetimeIndex(close.index.values)
        price_data[ticker] = close.rename(ticker)

        time.sleep(0.15)  # gentle pacing to avoid rate-limit on cloud

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
    }
    try:
        info = yf.Ticker(ticker, session=_SESSION).info
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice")
            or info.get("previousClose")
        )
        beta = info.get("beta") or info.get("beta3Year")
        div_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        return {
            "current_price":  float(price) if price is not None else None,
            "market_cap":     info.get("marketCap"),
            "beta":           float(beta) if beta is not None else None,
            "sector":         info.get("sector") or info.get("categoryName"),
            "dividend_yield": float(div_yield) if div_yield is not None else None,
            "week_52_high":   info.get("fiftyTwoWeekHigh"),
            "week_52_low":    info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return empty
