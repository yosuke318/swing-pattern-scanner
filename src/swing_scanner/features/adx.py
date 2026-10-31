"""ADX14 と +DI / -DI（基本設計書 4.3: ADX >= 閾値 かつ +DI > -DI）。

TradingView の ta.dmi(14, 14) と同じ定義:
  上昇幅 up = 高値 − 前日高値、下落幅 down = 前日安値 − 安値
  +DM = up   （up > down かつ up > 0 のとき。それ以外は 0）
  -DM = down （down > up かつ down > 0 のとき。それ以外は 0）
  +DI = 100 × RMA(+DM) ÷ RMA(TR)、-DI も同様
  DX  = 100 × |+DI − -DI| ÷ (+DI + -DI)
  ADX = RMA(DX)
RMA は ATR と同じ Wilder の平滑化。TR と ±DM は前日の値が要るので、先頭行は NaN にして
3つの平滑化の開始位置を揃える（ATR14 とは先頭行の扱いだけが違う）。

ADX はトレンドの強さだけを示し、向きは +DI / -DI の大小で見る。値は 14 行目から +DI / -DI、
27 行目から ADX14 が出る（それより前は NaN）。
"""

import math

import numpy as np
import pandas as pd

from swing_scanner.features.atr import true_range
from swing_scanner.features.smoothing import wilder_smoothing

ADX_PERIOD = 14


def directional_movement(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(+DM, -DM)。先頭行は NaN。"""
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    return plus, minus


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    # 分母が 0 のとき（値動きが全く無い期間）は 0 とする
    return (
        (numerator / denominator.replace(0, np.nan))
        .where(denominator != 0, 0.0)
        .where(denominator.notna())
    )


def dmi(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(+DI, -DI, ADX) を返す。"""
    tr = true_range(high, low, close)
    tr.iloc[:1] = math.nan
    plus_dm, minus_dm = directional_movement(high, low)
    tr_rma = wilder_smoothing(tr, period)
    plus_di = 100 * _ratio(wilder_smoothing(plus_dm, period), tr_rma)
    minus_di = 100 * _ratio(wilder_smoothing(minus_dm, period), tr_rma)
    dx = 100 * _ratio((plus_di - minus_di).abs(), plus_di + minus_di)
    return plus_di, minus_di, wilder_smoothing(dx, period)


def add_adx(df: pd.DataFrame) -> pd.DataFrame:
    """adx14 / plus_di / minus_di を足す。"""
    out = df.copy()
    out["plus_di"], out["minus_di"], out["adx14"] = dmi(out["high"], out["low"], out["close"])
    return out
