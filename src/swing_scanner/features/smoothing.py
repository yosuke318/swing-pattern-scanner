"""Wilder の平滑化（RMA）。ATR と ADX（[2-3]）で共通に使う。"""

import numpy as np
import pandas as pd


def wilder_smoothing(values: pd.Series, period: int) -> pd.Series:
    """Wilder の平滑化。TradingView の ta.rma と同じ定義。

    先頭の NaN を飛ばし、最初の period 個の単純平均を初期値にして、以降は
        RMA(t) = (RMA(t-1) × (period − 1) + x(t)) / period
    で更新する。初期値より前は NaN。途中に NaN が来たら、それ以降はすべて NaN にする
    （値を推測して埋めない。入力の欠損は [1-6] の検証で弾く前提）。
    """
    x = values.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) == 0:
        return pd.Series(out, index=values.index)
    start = valid[0]
    seed_end = start + period
    if seed_end > len(x) or np.isnan(x[start:seed_end]).any():
        return pd.Series(out, index=values.index)
    prev = x[start:seed_end].mean()
    out[seed_end - 1] = prev
    for i in range(seed_end, len(x)):
        prev = (prev * (period - 1) + x[i]) / period
        out[i] = prev
    return pd.Series(out, index=values.index)
