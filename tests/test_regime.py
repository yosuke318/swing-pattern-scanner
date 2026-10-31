import math

import numpy as np
import pandas as pd
import pytest
from test_features import assert_series, ohlcv

from swing_scanner.features.adx import add_adx
from swing_scanner.features.atr import add_atr
from swing_scanner.features.moving_average import add_moving_averages
from swing_scanner.features.regime import (
    add_regimes,
    apply_hysteresis,
    trading_value_20d,
    trend_by_adx,
    trend_by_ma,
    trend_by_structure,
    trend_by_vote,
    vol_regime,
)
from swing_scanner.features.swing import add_swings

nan = math.nan
NA = pd.NA
UP, RANGE, DOWN = "uptrend", "range", "downtrend"


def labels(values: list) -> pd.Series:
    return pd.Series(values, dtype="string")


def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return add_swings(add_adx(add_atr(add_moving_averages(df))))


def test_vol_regime_boundaries() -> None:
    pct = pd.Series([nan, 10, 20, 20.1, 79.9, 80, 95])
    expected = [NA, "low", "low", "normal", "normal", "high", "high"]
    assert vol_regime(pct).tolist() == expected


def test_vol_regime_distribution_is_about_20_60_20() -> None:
    frames = [add_atr(ohlcv(1250, seed=s)) for s in range(10)]
    shares = pd.concat([vol_regime(f["atr252_percentile"]) for f in frames]).value_counts(
        normalize=True
    )
    assert shares["low"] == pytest.approx(0.2, abs=0.05)
    assert shares["normal"] == pytest.approx(0.6, abs=0.05)
    assert shares["high"] == pytest.approx(0.2, abs=0.05)


def test_trend_by_adx() -> None:
    adx = pd.Series([25.0, 25, 25, 15, nan])
    plus = pd.Series([30.0, 10, 20, 30, 30])
    minus = pd.Series([10.0, 30, 20, 10, 10])
    assert trend_by_adx(adx, plus, minus).tolist() == [UP, DOWN, RANGE, RANGE, NA]


def test_trend_by_structure() -> None:
    structure = labels(["uptrend", "downtrend", "unclear", NA])
    assert trend_by_structure(structure).tolist() == [UP, DOWN, RANGE, NA]


def test_trend_by_ma() -> None:
    close = pd.Series([110.0, 90, 105, 100])
    ma25 = pd.Series([105.0, 95, 95, nan])
    ma75 = pd.Series([100.0, 100, 100, 100])
    assert trend_by_ma(close, ma25, ma75).tolist() == [UP, DOWN, RANGE, NA]


def test_trend_by_vote() -> None:
    adx = labels([UP, UP, UP, DOWN, UP, UP])
    structure = labels([UP, UP, RANGE, DOWN, RANGE, NA])
    ma = labels([RANGE, DOWN, RANGE, DOWN, RANGE, UP])
    # 2票で逆向き無し / 2票でも逆向きあり / 1票だけ / 3票 / 1票 / NA を含む
    assert trend_by_vote([adx, structure, ma]).tolist() == [UP, RANGE, RANGE, DOWN, RANGE, NA]


def test_hysteresis_waits_for_consecutive_days() -> None:
    raw = labels([UP, UP, RANGE, UP, RANGE, RANGE, RANGE, NA, DOWN])
    assert apply_hysteresis(raw, 2).tolist() == [UP, UP, UP, UP, UP, RANGE, RANGE, NA, DOWN]
    assert apply_hysteresis(raw, 1).equals(raw)
    with pytest.raises(ValueError):
        apply_hysteresis(raw, 0)


def test_trading_value_20d_prefers_api_value() -> None:
    df = pd.DataFrame({"close": [10.0] * 25, "volume": [100] * 25, "trading_value": [2000.0] * 25})
    assert_series(trading_value_20d(df), [nan] * 19 + [2000.0] * 6)
    assert_series(trading_value_20d(df.drop(columns="trading_value")), [nan] * 19 + [1000.0] * 6)


def test_add_regimes_on_pipeline() -> None:
    out = add_regimes(pipeline(ohlcv(300)))
    last = out.iloc[-1]
    assert last["vol_regime"] in {"low", "normal", "high"}
    assert last["trend_regime"] in {UP, RANGE, DOWN}
    assert last["trading_value_20d"] > 0
    # 252日パーセンタイルがそろう前は vol_regime が NA
    assert out["vol_regime"].iloc[:264].isna().all()


@pytest.mark.parametrize("method", ["adx", "structure", "ma", "vote"])
def test_each_trend_method_gives_all_three_labels(method: str) -> None:
    trend = add_regimes(pipeline(ohlcv(1250, seed=8)), trend_method=method)["trend_regime"]
    assert set(trend.dropna()) == {UP, RANGE, DOWN}


def test_invalid_trend_method() -> None:
    with pytest.raises(ValueError):
        add_regimes(pipeline(ohlcv(100)), trend_method="other")


def test_short_history_is_na_without_error() -> None:
    out = add_regimes(pipeline(ohlcv(30)))
    assert out["vol_regime"].isna().all()
    assert out["trend_regime"].isna().all()


def test_values_do_not_depend_on_future_rows() -> None:
    df = ohlcv(400, seed=9)
    full = add_regimes(pipeline(df), hysteresis_days=3)
    for cut in (280, 333, 390):
        head = add_regimes(pipeline(df.iloc[:cut]), hysteresis_days=3)
        pd.testing.assert_frame_equal(head, full.iloc[:cut], obj=f"cut={cut}")
    assert np.isfinite(full["trading_value_20d"].iloc[-1])
