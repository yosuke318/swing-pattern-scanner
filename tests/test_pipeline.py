import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_features import assert_series, ohlcv

from swing_scanner.config import FeatureSettings, load_config
from swing_scanner.features import EXTRA_COLUMNS, FEATURE_COLUMNS, compute_features
from swing_scanner.features.relative import relative_strength, volume_ratio

nan = math.nan
CONFIG = load_config(Path(__file__).resolve().parents[1] / "config")
SETTINGS = CONFIG.settings.features
LOOKBACK = CONFIG.settings.fetch.lookback_days
BENCH = ohlcv(1250, seed=999)

# 基本設計書 3.6 の列
DESIGN_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
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


# --- 出来高比率・相対強度（[2-5]） --------------------------------------------------


def test_volume_ratio_uses_previous_20_days() -> None:
    volume = pd.Series([100] * 20 + [300, 100])
    # 21 日目の分母は前日までの20日（すべて100）。22 日目の分母には 300 が入る
    assert_series(volume_ratio(volume), [nan] * 20 + [3.0, 100 / 110])


def test_volume_ratio_is_nan_when_no_trading() -> None:
    assert math.isnan(volume_ratio(pd.Series([0] * 20 + [100])).iloc[-1])


def test_relative_strength_is_return_difference() -> None:
    dates = pd.Series(pd.bdate_range("2026-01-05", periods=21).date)
    close = pd.Series([100.0] * 20 + [110.0])
    bench = pd.DataFrame({"date": dates, "close": [200.0] * 20 + [210.0]})
    assert relative_strength(dates, close, bench).iloc[-1] == pytest.approx(0.05)


def test_relative_strength_missing_benchmark_day_is_nan() -> None:
    dates = pd.Series(pd.bdate_range("2026-01-05", periods=30).date)
    close = pd.Series(np.linspace(100, 130, 30))
    bench = pd.DataFrame({"date": dates, "close": np.linspace(200, 230, 30)}).drop(index=5)
    rs = relative_strength(dates, close, bench)
    # 指数に無い 5 日目と、それを20日前として参照する 25 日目が NaN
    assert rs.isna().tolist() == [True] * 20 + [False] * 5 + [True] + [False] * 4


def test_benchmark_none_makes_only_relative_strength_nan() -> None:
    out = compute_features(ohlcv(300), None, SETTINGS)
    assert out["relative_strength_20d"].isna().all()
    assert out["volume_ratio_20d"].notna().iloc[20:].all()
    assert out["trend_regime"].notna().iloc[-1]


# --- 出力の形 ---------------------------------------------------------------------


def test_output_columns_match_design() -> None:
    out = compute_features(ohlcv(300), BENCH, SETTINGS)
    assert FEATURE_COLUMNS == DESIGN_COLUMNS
    assert list(out.columns[: len(DESIGN_COLUMNS)]) == DESIGN_COLUMNS
    assert list(out.columns) == DESIGN_COLUMNS + EXTRA_COLUMNS
    assert len(out) == 300


def test_extra_input_columns_are_kept() -> None:
    df = ohlcv(60).assign(trading_value=1e9)
    out = compute_features(df, None, SETTINGS)
    assert out.columns[-1] == "trading_value"
    assert out["trading_value_20d"].iloc[-1] == pytest.approx(1e9)


def test_input_is_not_modified_and_index_is_reset() -> None:
    df = ohlcv(300).set_index(pd.RangeIndex(1000, 1300))
    before = df.copy()
    out = compute_features(df, BENCH, SETTINGS)
    pd.testing.assert_frame_equal(df, before)
    assert out.index.equals(pd.RangeIndex(300))


def test_settings_change_the_calculation() -> None:
    other = SETTINGS.model_copy(update={"golden_cross_pair": (5, 25), "trend_method": "adx"})
    a = compute_features(ohlcv(300), BENCH, SETTINGS)
    b = compute_features(ohlcv(300), BENCH, other)
    assert not a["days_since_golden_cross"].equals(b["days_since_golden_cross"])
    assert not a["trend_regime"].equals(b["trend_regime"])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns="volume"), "volume"),
        (lambda d: d.iloc[::-1], "昇順"),
        (lambda d: pd.concat([d, d.iloc[-1:]]), "重複"),
    ],
)
def test_invalid_input(mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        compute_features(mutate(ohlcv(50)), None, SETTINGS)


def test_short_history_does_not_raise() -> None:
    out = compute_features(ohlcv(10), BENCH, SETTINGS)
    assert out["ma200"].isna().all() and out["vol_regime"].isna().all()


# --- 入力の長さ（日次の約300営業日とバックテストの5年）で同じ日の値が揃うこと -----------------


@pytest.mark.parametrize("seed", range(10))
def test_daily_and_backtest_inputs_agree_on_recent_days(seed: int) -> None:
    df = ohlcv(1250, seed=seed)
    long = compute_features(df, BENCH, SETTINGS).iloc[-20:].reset_index(drop=True)
    short = compute_features(df.iloc[-LOOKBACK:], BENCH, SETTINGS).iloc[-20:]
    short = short.reset_index(drop=True)

    # Wilder 平滑は初期値の影響が残るが、約300営業日あれば相対誤差 1e-6 未満
    smooth = [c for c in long.columns if c.startswith(("ma", "atr14", "close_minus"))]
    smooth += ["adx14", "plus_di", "minus_di", "trading_value_20d", "relative_strength_20d"]
    for col in smooth:
        np.testing.assert_allclose(short[col], long[col], rtol=1e-6, err_msg=col)

    # 252日の順位は、窓の古い側の ATR に初期値の影響がまだ残るので数ポイントずれうる
    assert (short["atr252_percentile"] - long["atr252_percentile"]).abs().max() <= 3

    # 判定に使う区分・スイングは一致する
    for col in [
        "vol_regime",
        "trend_regime",
        "hh_hl_structure",
        "swing_high_price",
        "swing_low_price",
        "prior_swing_low_price",
        "golden_cross_flag",
    ]:
        pd.testing.assert_series_equal(short[col], long[col], obj=col)

    # 300営業日より前の GD は短い入力では見えず inf になる。どちらも「X日以内」には当たらない
    a, b = short["days_since_golden_cross"], long["days_since_golden_cross"]
    assert ((a == b) | ((a > 200) & (b > 200))).all()


# --- 先読みしないこと --------------------------------------------------------------


def test_values_do_not_depend_on_future_rows() -> None:
    df = ohlcv(400, seed=11)
    full = compute_features(df, BENCH, SETTINGS)
    for cut in range(270, 400, 13):
        head = compute_features(df.iloc[:cut], BENCH, SETTINGS)
        pd.testing.assert_frame_equal(head, full.iloc[:cut], obj=f"cut={cut}")


def test_benchmark_future_rows_do_not_matter() -> None:
    df = ohlcv(300, seed=12)
    last_date = df["date"].iloc[-1]
    cut_bench = BENCH[BENCH["date"] <= last_date]
    a = compute_features(df, BENCH, SETTINGS)
    b = compute_features(df, cut_bench, SETTINGS)
    pd.testing.assert_frame_equal(a, b)


def test_feature_settings_reject_reversed_cross_pair() -> None:
    data = SETTINGS.model_dump() | {"golden_cross_pair": (75, 25)}
    with pytest.raises(ValueError, match="golden_cross_pair"):
        FeatureSettings(**data)
