"""ATR14 / atr14_pct / atr252_percentile（要件4: 閾値は各銘柄の過去ボラティリティ基準で決める）。

- True Range = max(高値 − 安値, |高値 − 前日終値|, |安値 − 前日終値|)。窓開けを含む。
  前日終値が無い先頭行は 高値 − 安値。ストップ高・安の張り付き（高値 = 安値）でも、
  前日終値との差で値幅が入る
- ATR14 は Wilder の平滑化（smoothing.wilder_smoothing）。TradingView の ta.atr(14) と同じ定義。
  平滑化方式を変えるとバックテスト結果が再現できなくなるので、ATR_SMOOTHING として記録する
- atr14_pct = ATR14 ÷ 終値（比率。0.023 なら 2.3%）
- atr252_percentile = 当日を含む直近252営業日の atr14_pct の中での当日の順位（0〜100、
  同値は平均順位）。252日そろわない行は NaN（上場から日が浅い銘柄は評価不能にする）
"""

import pandas as pd

from swing_scanner.features.smoothing import wilder_smoothing

ATR_PERIOD = 14
ATR_SMOOTHING = "wilder"
PERCENTILE_WINDOW = 252


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    tr = ranges.max(axis=1, skipna=False)
    # 先頭行は前日終値が無いので 高値 − 安値
    tr.iloc[:1] = (high - low).iloc[:1]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    return wilder_smoothing(true_range(high, low, close), period)


def rolling_percentile(values: pd.Series, window: int = PERCENTILE_WINDOW) -> pd.Series:
    """当日を含む直近 window 行の中での当日の順位（0〜100）。window 行そろわなければ NaN。"""
    return values.rolling(window, min_periods=window).rank(method="average", pct=True) * 100


def close_minus_ma_atr_ratio(close: pd.Series, ma: pd.Series, atr14: pd.Series) -> pd.Series:
    """(終値 − MA) ÷ ATR14。パターン条件 close_minus_ma<n>_atr_ratio の値。"""
    return (close - ma) / atr14


def add_atr(df: pd.DataFrame) -> pd.DataFrame:
    """atr14 / atr14_pct / atr252_percentile を足す。"""
    out = df.copy()
    out["atr14"] = atr(out["high"], out["low"], out["close"])
    out["atr14_pct"] = out["atr14"] / out["close"]
    out["atr252_percentile"] = rolling_percentile(out["atr14_pct"])
    return out
