"""移動平均とゴールデンクロス（基本設計書 4.1）。

GD発生 = 前日 MA_short <= MA_long かつ 当日 MA_short > MA_long

どの MA の組で GD とするかはパターンによって変えたいので、関数は組を引数で受け取る。
指標の DataFrame の golden_cross_flag / days_since_golden_cross は GOLDEN_CROSS_PAIR の組で作る。
"""

import math

import numpy as np
import pandas as pd

MA_PERIODS = (5, 25, 75, 200)
# 既定のゴールデンクロスの組（短期, 長期）。パターンから組を選べるようにするのは [3-1] [3-2]
GOLDEN_CROSS_PAIR = (25, 75)


def sma(close: pd.Series, period: int) -> pd.Series:
    """単純移動平均。period 日そろわない行は NaN。"""
    return close.rolling(period, min_periods=period).mean()


def _cross(above: pd.Series, below: pd.Series) -> pd.Series:
    """前日 above <= below かつ当日 above > below。4つの値のどれかが NaN なら NA。"""
    prev_above, prev_below = above.shift(1), below.shift(1)
    crossed = (prev_above <= prev_below) & (above > below)
    known = above.notna() & below.notna() & prev_above.notna() & prev_below.notna()
    return crossed.astype("boolean").where(known, pd.NA)


def golden_cross(ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    return _cross(ma_short, ma_long)


def dead_cross(ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    """前日 MA_short >= MA_long かつ当日 MA_short < MA_long。"""
    return _cross(ma_long, ma_short)


def days_since(flag: pd.Series) -> pd.Series:
    """直近で flag が True になった行からの経過行数（営業日数）。True の当日は 0。

    - flag が NA の行（MA の期間不足）は NaN（評価不能）
    - 判定できる状態になってから一度も True が無い行は inf。GD は両方の MA がそろって
      初めて起こりうるので、「まだ GD が起きていない」と確定できる。ただし入力の先頭で
      切れた期間（取得日数 N より前）の GD は数えない
    - 途中で NA が挟まったら、そこで数え直す（NA の後は再び True が来るまで inf）
    """
    out = np.full(len(flag), np.nan)
    count: float | None = None
    for i, value in enumerate(flag.to_numpy(dtype=object, na_value=None)):
        if value is None:
            count = None
            continue
        if value:
            count = 0
        elif count is None:
            count = math.inf
        else:
            count += 1
        out[i] = count
    return pd.Series(out, index=flag.index)


def add_moving_averages(
    df: pd.DataFrame,
    periods: tuple[int, ...] = MA_PERIODS,
    cross_pair: tuple[int, int] = GOLDEN_CROSS_PAIR,
) -> pd.DataFrame:
    """ma<n> 列と golden_cross_flag / dead_cross_flag / days_since_golden_cross を足す。"""
    short, long = cross_pair
    if short >= long:
        raise ValueError(f"ゴールデンクロスの組は (短期, 長期) で指定してください: {cross_pair}")
    out = df.copy()
    for period in sorted({*periods, short, long}):
        out[f"ma{period}"] = sma(out["close"], period)
    out["golden_cross_flag"] = golden_cross(out[f"ma{short}"], out[f"ma{long}"])
    out["dead_cross_flag"] = dead_cross(out[f"ma{short}"], out[f"ma{long}"])
    out["days_since_golden_cross"] = days_since(out["golden_cross_flag"])
    return out
