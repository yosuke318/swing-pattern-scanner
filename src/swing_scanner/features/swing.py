"""スイング高値・安値と HH/HL 構造（基本設計書 4.2、Dow 理論）。

スイング高値: 高値が前 N 日のどれよりも高く、後 N 日のどれ以上でもある日（後ろは同値を許す。
同じ高値が並んだ天井では最初の日だけがスイングになる）。スイング安値も同様。

スイングの確定には後ろの N 日が要るので、i 日目のスイングは i+N 日目に初めて使える。各行の
値は、その日までに確定したスイングだけで決める（先読みしない）。したがってスイング列は
実際の天井・底から N 営業日遅れて切り替わる。

N は固定値で始め（SWING_WINDOW）、引数で変えてバックテストで比較する。ATR 基準の可変 N は
比較の結果を見てから検討する。

hh_hl_structure は、直近 pairs 個のスイング高値と安値を見て
  uptrend:   高値も安値も、古い順に切り上がっている（Higher High かつ Higher Low）
  downtrend: 高値も安値も切り下がっている（Lower High かつ Lower Low）
  unclear:   それ以外
確定したスイングが pairs 個に満たなければ NA（評価不能）。
高値と安値が交互に現れることは前提にしていない（高値が2つ続けて確定することもある）。
"""

import numpy as np
import pandas as pd

SWING_WINDOW = 3
STRUCTURE_PAIRS = 2


def pivot_highs(high: pd.Series, window: int = SWING_WINDOW) -> pd.Series:
    """スイング高値の日に True（その日の位置に立てる。確定するのは window 日後）。"""
    left = high.shift(1).rolling(window, min_periods=window).max()
    right = high[::-1].shift(1).rolling(window, min_periods=window).max()[::-1]
    return (high > left) & (high >= right)


def pivot_lows(low: pd.Series, window: int = SWING_WINDOW) -> pd.Series:
    left = low.shift(1).rolling(window, min_periods=window).min()
    right = low[::-1].shift(1).rolling(window, min_periods=window).min()[::-1]
    return (low < left) & (low <= right)


def confirmed_swings(
    prices: pd.Series, is_pivot: pd.Series, window: int, count: int
) -> list[pd.Series]:
    """各行の時点で確定している直近 count 個のスイングの価格。[直近, 1つ前, ...] の順。"""
    # i 日目のスイングは i+window 日目に確定する
    confirmed = prices.where(is_pivot).shift(window).to_numpy(dtype=float)
    out = np.full((count, len(confirmed)), np.nan)
    recent: list[float] = []
    for t, price in enumerate(confirmed):
        if not np.isnan(price):
            recent = [price, *recent[: count - 1]]
        out[: len(recent), t] = recent
    return [pd.Series(row, index=prices.index) for row in out]


def _rising(swings: list[pd.Series]) -> pd.Series:
    """直近から古い方へ並んだスイングが、古い順に切り上がっているか。"""
    result = pd.Series(True, index=swings[0].index)
    for newer, older in zip(swings, swings[1:], strict=False):
        result &= newer > older
    return result


def _falling(swings: list[pd.Series]) -> pd.Series:
    result = pd.Series(True, index=swings[0].index)
    for newer, older in zip(swings, swings[1:], strict=False):
        result &= newer < older
    return result


def hh_hl_structure(highs: list[pd.Series], lows: list[pd.Series]) -> pd.Series:
    known = pd.concat([*highs, *lows], axis=1).notna().all(axis=1)
    up = _rising(highs) & _rising(lows)
    down = _falling(highs) & _falling(lows)
    labels = np.where(up, "uptrend", np.where(down, "downtrend", "unclear"))
    return pd.Series(labels, index=known.index, dtype="string").where(known, pd.NA)


def add_swings(
    df: pd.DataFrame, window: int = SWING_WINDOW, pairs: int = STRUCTURE_PAIRS
) -> pd.DataFrame:
    """swing_high_price / prior_swing_high_price / swing_low_price / prior_swing_low_price /
    swing_low_higher_than_prior / hh_hl_structure を足す。"""
    if window < 1 or pairs < 2:
        raise ValueError(f"window は1以上、pairs は2以上にしてください: {window}, {pairs}")
    out = df.copy()
    highs = confirmed_swings(out["high"], pivot_highs(out["high"], window), window, pairs)
    lows = confirmed_swings(out["low"], pivot_lows(out["low"], window), window, pairs)
    out["swing_high_price"], out["prior_swing_high_price"] = highs[0], highs[1]
    out["swing_low_price"], out["prior_swing_low_price"] = lows[0], lows[1]
    known = lows[0].notna() & lows[1].notna()
    out["swing_low_higher_than_prior"] = (lows[0] > lows[1]).astype("boolean").where(known, pd.NA)
    out["hh_hl_structure"] = hh_hl_structure(highs, lows)
    return out
