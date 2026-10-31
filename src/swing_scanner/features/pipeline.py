"""指標計算の入口（基本設計書 5.2 の手順5）。日次処理とバックテストで同じ関数を使う。

日次処理は約300営業日を入力して最終行だけを判定に使い、バックテストは最大5年を入力して全行を
使う。入力の長さが違うだけで関数は同じ。
"""

import pandas as pd

from swing_scanner.config import FeatureSettings
from swing_scanner.features.adx import add_adx
from swing_scanner.features.atr import add_atr, close_minus_ma_atr_ratio
from swing_scanner.features.moving_average import MA_PERIODS, add_moving_averages
from swing_scanner.features.regime import add_regimes
from swing_scanner.features.relative import add_relative
from swing_scanner.features.swing import add_swings

INPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# 基本設計書 3.6 の列
FEATURE_COLUMNS = [
    *INPUT_COLUMNS,
    "ma5",
    "ma25",
    "ma75",
    "ma200",
    "atr14",
    "atr14_pct",
    "atr252_percentile",
    "adx14",
    "plus_di",
    "minus_di",
    "golden_cross_flag",
    "days_since_golden_cross",
    "swing_high_price",
    "swing_low_price",
    "prior_swing_low_price",
    "hh_hl_structure",
    "volume_ratio_20d",
    "relative_strength_20d",
    "trading_value_20d",
    "vol_regime",
    "trend_regime",
]

# 3.6 には無いが、判定やログで使う列
EXTRA_COLUMNS = [
    "dead_cross_flag",
    "prior_swing_high_price",
    "swing_low_higher_than_prior",
    *(f"close_minus_ma{n}_atr_ratio" for n in MA_PERIODS),
]


def _check_input(ohlcv: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in INPUT_COLUMNS if c not in ohlcv]
    if missing:
        raise ValueError(f"{name} に列がありません: {missing}")
    dates = pd.Series(ohlcv["date"])
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"{name} の date は重複なしの昇順にしてください")
    return ohlcv.reset_index(drop=True)


def compute_features(
    ohlcv: pd.DataFrame, benchmark: pd.DataFrame | None, settings: FeatureSettings
) -> pd.DataFrame:
    """1銘柄の日足から指標の DataFrame（全日付ぶん）を作る。

    ohlcv: 列 date / open / high / low / close / volume（任意で trading_value）、日付の昇順。
    benchmark: 指数ETFの日足（date / close があればよい）。取得できなかった日は None を渡し、
      relative_strength_20d を NaN（評価不能）にする。
    期間不足の列は NaN / NA のまま返す。各行の値はその日までのデータだけで決まる。
    """
    df = _check_input(ohlcv, "ohlcv")
    if benchmark is not None:
        benchmark = _check_input(benchmark, "benchmark")
    cross_pair = settings.golden_cross_pair
    df = add_moving_averages(df, periods=MA_PERIODS, cross_pair=cross_pair)
    df = add_atr(df)
    df = add_adx(df)
    df = add_swings(df, window=settings.swing_window, pairs=settings.structure_pairs)
    df = add_relative(df, benchmark)
    df = add_regimes(
        df, trend_method=settings.trend_method, hysteresis_days=settings.hysteresis_days
    )
    for n in MA_PERIODS:
        df[f"close_minus_ma{n}_atr_ratio"] = close_minus_ma_atr_ratio(
            df["close"], df[f"ma{n}"], df["atr14"]
        )
    passthrough = [c for c in df.columns if c not in FEATURE_COLUMNS + EXTRA_COLUMNS]
    return df[FEATURE_COLUMNS + EXTRA_COLUMNS + passthrough]
