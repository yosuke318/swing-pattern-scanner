import math

import numpy as np
import pandas as pd
import pytest

from swing_scanner.features.atr import (
    ATR_PERIOD,
    PERCENTILE_WINDOW,
    add_atr,
    close_minus_ma_atr_ratio,
    rolling_percentile,
    true_range,
)
from swing_scanner.features.moving_average import (
    add_moving_averages,
    days_since,
    dead_cross,
    golden_cross,
    sma,
)
from swing_scanner.features.smoothing import wilder_smoothing

NA = pd.NA
nan = math.nan
inf = math.inf


def ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    """ランダムウォークの日足（窓開けを含む）。"""
    rng = np.random.default_rng(seed)
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n).date,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n),
        }
    )


def assert_series(actual: pd.Series, expected: list) -> None:
    np.testing.assert_allclose(actual.to_numpy(dtype=float), np.array(expected, dtype=float))


def flags(values: list) -> pd.Series:
    return pd.Series(values, dtype="boolean")


# --- 移動平均・ゴールデンクロス --------------------------------------------------


def test_sma_is_nan_until_period_filled() -> None:
    assert_series(sma(pd.Series([1.0, 2, 3, 4, 5]), 3), [nan, nan, 2, 3, 4])


def test_golden_cross_definition() -> None:
    # 前日 short <= long（等しい場合も含む）かつ当日 short > long
    short = pd.Series([1.0, 1, 2, 2, 1, 1, 3])
    long = pd.Series([2.0, 1, 1, 1, 2, 1, 2])
    expected = [NA, False, True, False, False, False, True]
    assert golden_cross(short, long).tolist() == expected


def test_dead_cross_definition() -> None:
    short = pd.Series([2.0, 1, 1, 2, 2])
    long = pd.Series([1.0, 1, 2, 1, 1])
    assert dead_cross(short, long).tolist() == [NA, False, True, False, False]


def test_cross_is_na_when_any_ma_missing() -> None:
    short = pd.Series([nan, 1.0, 2, 3])
    long = pd.Series([nan, nan, 1.0, 2])
    assert golden_cross(short, long).tolist() == [NA, NA, NA, False]


def test_days_since_counts_rows_and_distinguishes_never_from_unknown() -> None:
    flag = flags([NA, NA, False, False, True, False, False, True, False, NA, False])
    assert_series(days_since(flag), [nan, nan, inf, inf, 0, 1, 2, 0, 1, nan, inf])


def test_add_moving_averages_hand_calculated() -> None:
    df = pd.DataFrame({"close": [10.0, 9, 8, 7, 8, 9, 10, 11]})
    out = add_moving_averages(df, periods=(2, 4), cross_pair=(2, 4))
    assert_series(out["ma2"], [nan, 9.5, 8.5, 7.5, 7.5, 8.5, 9.5, 10.5])
    assert_series(out["ma4"], [nan, nan, nan, 8.5, 8.0, 8.0, 8.5, 9.5])
    assert out["golden_cross_flag"].tolist() == [NA] * 4 + [False, True, False, False]
    assert_series(out["days_since_golden_cross"], [nan] * 4 + [inf, 0, 1, 2])
    assert "close" in out and "ma2" not in df  # 入力は書き換えない


def test_add_moving_averages_default_columns() -> None:
    out = add_moving_averages(ohlcv(300))
    for col in ["ma5", "ma25", "ma75", "ma200", "golden_cross_flag", "days_since_golden_cross"]:
        assert col in out
    assert out["ma200"].first_valid_index() == 199
    # 既定の組は 25 / 75。MA75 が前日・当日ともにそろう 76 行目から判定できる
    assert out["golden_cross_flag"].first_valid_index() == 75


def test_short_history_gives_nan_without_error() -> None:
    out = add_moving_averages(ohlcv(60))
    assert out["ma75"].isna().all() and out["ma200"].isna().all()
    assert out["golden_cross_flag"].isna().all()
    assert out["days_since_golden_cross"].isna().all()


def test_cross_pair_must_be_short_then_long() -> None:
    with pytest.raises(ValueError):
        add_moving_averages(ohlcv(10), cross_pair=(75, 25))


# --- ATR ---------------------------------------------------------------------


def test_true_range_includes_gaps() -> None:
    high = pd.Series([10.0, 12, 15, 16, 13])
    low = pd.Series([8.0, 9, 14, 16, 12])
    close = pd.Series([9.0, 11, 14.5, 16, 12.5])
    # 先頭は高値−安値 / 通常 / 窓開け上昇 / ストップ高張り付き（高値=安値）/ 窓開け下落
    assert_series(true_range(high, low, close), [2, 3, 4, 1.5, 4])


def test_wilder_smoothing_hand_calculated() -> None:
    out = wilder_smoothing(pd.Series([2.0, 4, 6, 8, 10]), 3)
    assert_series(out, [nan, nan, 4, 16 / 3, 62 / 9])


def test_wilder_smoothing_skips_leading_nan_and_stops_at_gap() -> None:
    assert_series(wilder_smoothing(pd.Series([nan, 2.0, 4, 6, 9]), 3), [nan, nan, nan, 4, 17 / 3])
    assert_series(wilder_smoothing(pd.Series([2.0, 4, 6, nan, 9]), 3), [nan, nan, 4, nan, nan])
    assert wilder_smoothing(pd.Series([2.0, 4]), 3).isna().all()


def test_atr_matches_ewm_formulation() -> None:
    # 別の書き方で突き合わせる: 初期値を単純平均にした alpha = 1/14 の指数平滑と一致する
    df = add_atr(ohlcv(300, seed=1))
    tr = true_range(df["high"], df["low"], df["close"])
    seeded = tr.iloc[ATR_PERIOD - 1 :].copy()
    seeded.iloc[0] = tr.iloc[:ATR_PERIOD].mean()
    expected = seeded.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    np.testing.assert_allclose(df["atr14"].iloc[ATR_PERIOD - 1 :], expected, rtol=1e-12)
    assert df["atr14"].iloc[: ATR_PERIOD - 1].isna().all()


def test_atr14_pct_is_ratio_to_close() -> None:
    df = add_atr(ohlcv(50))
    np.testing.assert_allclose(df["atr14_pct"], df["atr14"] / df["close"])


def test_rolling_percentile_hand_calculated() -> None:
    out = rolling_percentile(pd.Series([1.0, 2, 3, 2, 1]), window=3)
    assert_series(out, [nan, nan, 100, 50, 100 / 3])


def test_atr252_percentile_needs_full_window() -> None:
    df = add_atr(ohlcv(300))
    first = ATR_PERIOD - 1 + PERCENTILE_WINDOW - 1
    assert df["atr252_percentile"].first_valid_index() == first
    valid = df["atr252_percentile"].dropna()
    assert ((valid > 0) & (valid <= 100)).all()


def test_short_history_atr_percentile_is_nan_without_error() -> None:
    df = add_atr(ohlcv(200))
    assert df["atr14"].notna().iloc[ATR_PERIOD - 1 :].all()
    assert df["atr252_percentile"].isna().all()


def test_close_minus_ma_atr_ratio() -> None:
    ratio = close_minus_ma_atr_ratio(pd.Series([110.0]), pd.Series([100.0]), pd.Series([4.0]))
    assert_series(ratio, [2.5])


# --- 先読みしないこと --------------------------------------------------------------


def test_values_do_not_depend_on_future_rows() -> None:
    full = add_atr(add_moving_averages(ohlcv(400, seed=2)))
    cut = 320
    head = add_atr(add_moving_averages(ohlcv(400, seed=2).iloc[:cut]))
    cols = [c for c in head.columns if c not in ("date", "golden_cross_flag", "dead_cross_flag")]
    pd.testing.assert_frame_equal(head[cols], full[cols].iloc[:cut])
    assert head["golden_cross_flag"].equals(full["golden_cross_flag"].iloc[:cut])
