"""レジーム分類（基本設計書 4.4 / 4.5）と流動性指標。

vol_regime（設計書 4.4）: atr252_percentile が 20 以下なら low、80 以上なら high、
それ以外は normal。
銘柄自身の過去252日での順位なので、長い期間で見れば各銘柄で概ね 20% / 60% / 20% になる。

trend_regime（uptrend / range / downtrend）は設計書に定義が無いのでここで決める。どの定義に
するかで検知数が大きく変わるため、実データで該当日数を比べてから選べるよう、方式を切り替えられる:
  adx:       ADX14 >= 20 で +DI > -DI なら uptrend、-DI > +DI なら downtrend、それ以外は range
  structure: hh_hl_structure の uptrend / downtrend をそのまま使い、unclear は range
  ma:        終値 > MA25 > MA75 なら uptrend、終値 < MA25 < MA75 なら downtrend、それ以外は range
  vote:      上の3つのうち2つ以上が同じ向きで、逆向きが1つも無ければその向き。それ以外は range
既定は vote（3つの AND は厳しすぎて何も出ない恐れがあり、単独では1つの指標の癖をそのまま受ける）。
入力のどれかが NA の行は NA（評価不能）。

hysteresis_days を 2 以上にすると、新しい区分が連続でその日数続くまで切り替えない（境界の銘柄で
適用パターンが日替わりするのを抑える）。既定 1 は切り替えを遅らせない。

trading_value_20d: 20日平均売買代金。trading_value 列（API が返す売買代金）があればそれを使い、
無ければ 終値 × 出来高 で代用する。
"""

from typing import Literal

import numpy as np
import pandas as pd

TrendMethod = Literal["adx", "structure", "ma", "vote"]

VOL_LOW_PERCENTILE = 20
VOL_HIGH_PERCENTILE = 80
# トレンドありとみなす ADX14 の下限。暫定値で、バックテストで見直す
TREND_ADX_THRESHOLD = 20
TREND_METHOD: TrendMethod = "vote"
TRADING_VALUE_WINDOW = 20

_UP, _RANGE, _DOWN = "uptrend", "range", "downtrend"


def _labels(up: pd.Series, down: pd.Series, known: pd.Series, neutral: str) -> pd.Series:
    values = np.where(up, _UP, np.where(down, _DOWN, neutral))
    return pd.Series(values, index=up.index, dtype="string").where(known, pd.NA)


def vol_regime(atr252_percentile: pd.Series) -> pd.Series:
    low = atr252_percentile <= VOL_LOW_PERCENTILE
    high = atr252_percentile >= VOL_HIGH_PERCENTILE
    values = np.where(low, "low", np.where(high, "high", "normal"))
    return pd.Series(values, index=low.index, dtype="string").where(
        atr252_percentile.notna(), pd.NA
    )


def trend_by_adx(adx14: pd.Series, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    strong = adx14 >= TREND_ADX_THRESHOLD
    known = adx14.notna() & plus_di.notna() & minus_di.notna()
    return _labels(strong & (plus_di > minus_di), strong & (minus_di > plus_di), known, _RANGE)


def trend_by_structure(hh_hl_structure: pd.Series) -> pd.Series:
    return hh_hl_structure.replace("unclear", _RANGE).astype("string")


def trend_by_ma(close: pd.Series, ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    known = close.notna() & ma_short.notna() & ma_long.notna()
    up = (close > ma_short) & (ma_short > ma_long)
    down = (close < ma_short) & (ma_short < ma_long)
    return _labels(up, down, known, _RANGE)


def trend_by_vote(signals: list[pd.Series]) -> pd.Series:
    frame = pd.concat(signals, axis=1)
    ups = (frame == _UP).sum(axis=1)
    downs = (frame == _DOWN).sum(axis=1)
    known = frame.notna().all(axis=1)
    return _labels((ups >= 2) & (downs == 0), (downs >= 2) & (ups == 0), known, _RANGE)


def apply_hysteresis(labels: pd.Series, days: int) -> pd.Series:
    """新しい区分が days 日連続するまで前の区分を保つ。NA の行は NA にし、次の値から数え直す。"""
    if days < 1:
        raise ValueError(f"hysteresis_days は1以上にしてください: {days}")
    if days == 1:
        return labels
    out: list[object] = []
    current: str | None = None
    streak_label: str | None = None
    streak = 0
    for value in labels.to_numpy(dtype=object, na_value=None):
        if value is None:
            current, streak_label, streak = None, None, 0
            out.append(pd.NA)
            continue
        streak = streak + 1 if value == streak_label else 1
        streak_label = value
        if current is None or streak >= days:
            current = value
        out.append(current)
    return pd.Series(out, index=labels.index, dtype="string")


def trend_regime(df: pd.DataFrame, method: TrendMethod = TREND_METHOD) -> pd.Series:
    signals = {
        "adx": lambda: trend_by_adx(df["adx14"], df["plus_di"], df["minus_di"]),
        "structure": lambda: trend_by_structure(df["hh_hl_structure"]),
        "ma": lambda: trend_by_ma(df["close"], df["ma25"], df["ma75"]),
    }
    if method == "vote":
        return trend_by_vote([make() for make in signals.values()])
    if method not in signals:
        raise ValueError(f"trend_regime の方式が不正です: {method!r}")
    return signals[method]()


def trading_value_20d(df: pd.DataFrame) -> pd.Series:
    value = df["trading_value"] if "trading_value" in df else df["close"] * df["volume"]
    return (
        value.astype(float).rolling(TRADING_VALUE_WINDOW, min_periods=TRADING_VALUE_WINDOW).mean()
    )


def add_regimes(
    df: pd.DataFrame, trend_method: TrendMethod = TREND_METHOD, hysteresis_days: int = 1
) -> pd.DataFrame:
    """vol_regime / trend_regime / trading_value_20d を足す。

    入力には atr252_percentile（add_atr）、adx14 / plus_di / minus_di（add_adx）、
    hh_hl_structure（add_swings）、ma25 / ma75（add_moving_averages）が要る。
    """
    out = df.copy()
    out["vol_regime"] = apply_hysteresis(vol_regime(out["atr252_percentile"]), hysteresis_days)
    out["trend_regime"] = apply_hysteresis(trend_regime(out, trend_method), hysteresis_days)
    out["trading_value_20d"] = trading_value_20d(out)
    return out
