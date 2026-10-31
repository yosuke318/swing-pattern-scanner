"""取得データの検証（最小限。APIエラー・欠損・異常値の扱いは [1-6] で広げる）。

検証に通らない銘柄は判定から外し、警告に積む（基本設計書 5.3）。
"""

import datetime as dt

import pandas as pd

from swing_scanner.sources.base import OHLCV_COLUMNS


def validate_ohlcv(df: pd.DataFrame, target_date: dt.date) -> str | None:
    """問題があれば理由を返す。無ければ None。"""
    missing = [c for c in OHLCV_COLUMNS if c not in df]
    if missing:
        return f"列がありません: {missing}"
    if df.empty:
        return "データがありません"
    dates = pd.Series(df["date"])
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        return "日付の並びが不正です"
    last = dates.iloc[-1]
    if last != target_date:
        return f"{target_date} のデータがありません（最終 {last}）"
    values = df[OHLCV_COLUMNS[1:]]
    if values.isna().any().any():
        return "OHLCV に欠損があります"
    prices = df[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (df["high"] < df["low"]).any() or (df["volume"] < 0).any():
        return "OHLCV に異常値があります"
    return None
