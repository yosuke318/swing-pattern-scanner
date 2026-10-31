import math

import numpy as np
import pandas as pd
import pytest
from test_features import assert_series, ohlcv

from swing_scanner.features.swing import add_swings, pivot_highs, pivot_lows

nan = math.nan
NA = pd.NA


def bars(highs: list[float], lows: list[float] | None = None) -> pd.DataFrame:
    lows = lows if lows is not None else [h - 1 for h in highs]
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": [(h + lo) / 2 for h, lo in zip(highs, lows, strict=True)],
        }
    )


def test_pivot_high_needs_window_on_both_sides() -> None:
    high = pd.Series([1.0, 2, 5, 3, 2, 4, 6, 4])
    # 2 日目（5）は前後2日より高い。6 日目（6）は後ろが1日しか無いので確定しない
    assert pivot_highs(high, 2).tolist() == [False, False, True, False, False, False, False, False]


def test_flat_top_marks_only_first_day() -> None:
    high = pd.Series([1.0, 2, 5, 5, 3, 2, 1])
    assert pivot_highs(high, 2).tolist() == [False, False, True, False, False, False, False]


def test_pivot_low() -> None:
    low = pd.Series([5.0, 4, 1, 3, 4, 2, 3, 4])
    assert pivot_lows(low, 2).tolist() == [False, False, True, False, False, True, False, False]


def test_swing_is_used_only_after_confirmation() -> None:
    # 高値 5 は 2 日目。window=2 なので 4 日目から swing_high_price に入る。
    # 5 日目の高値 4 も前後2日より高く、7 日目に確定して入れ替わる
    out = add_swings(bars([1.0, 2, 5, 3, 2, 4, 3, 2]), window=2)
    assert_series(out["swing_high_price"], [nan, nan, nan, nan, 5, 5, 5, 4])


def test_prior_swings_and_structure_uptrend() -> None:
    # 安値 10 → 高値 20 → 安値 14 → 高値 24 と切り上がる波
    highs = [15.0, 13, 11, 13, 17, 20, 18, 16, 15, 18, 21, 24, 22, 20, 19]
    lows = [h - 1 for h in highs]
    lows[2], lows[8] = 10, 14
    out = add_swings(bars(highs, lows), window=2)
    last = out.iloc[-1]
    assert (last["swing_high_price"], last["prior_swing_high_price"]) == (24, 20)
    assert (last["swing_low_price"], last["prior_swing_low_price"]) == (14, 10)
    assert last["swing_low_higher_than_prior"]
    assert last["hh_hl_structure"] == "uptrend"
    # 高値 24（11日目）が確定する 13 日目より前は、高値が1つしか確定していないので NA
    assert out["hh_hl_structure"].iloc[12] is NA
    assert out["hh_hl_structure"].iloc[13] == "uptrend"


def test_structure_downtrend_and_unclear() -> None:
    highs = [25.0, 27, 30, 27, 24, 22, 24, 26, 24, 21, 18, 20, 22, 20, 18]
    lows = [h - 1 for h in highs]
    out = add_swings(bars(highs, lows), window=2)
    # 高値 30 → 26、安値 21 → 17 と切り下がる
    assert out.iloc[-1]["hh_hl_structure"] == "downtrend"

    # 三角保ち合い: 高値は 26 → 24 → 22 と切り下がり、安値は 12 → 14 と切り上がる
    highs2 = [20.0, 22, 26, 22, 20, 19, 20, 24, 21, 19, 18, 19, 22, 20, 19]
    lows2 = [15.0, 14, 16, 17, 12, 13, 14, 18, 17, 15, 14, 16, 18, 17, 16]
    out2 = add_swings(bars(highs2, lows2), window=2)
    last = out2.iloc[-1]
    assert (last["swing_high_price"], last["prior_swing_high_price"]) == (22, 24)
    assert (last["swing_low_price"], last["prior_swing_low_price"]) == (14, 12)
    assert out2.iloc[-1]["hh_hl_structure"] == "unclear"


def test_more_pairs_require_longer_staircase() -> None:
    # 安値が 10 → 14 → 12 → 16。直近2つは切り上がるが、3つでは切り上がっていない
    highs = [20.0, 18, 16, 18, 22, 20, 19, 21, 24, 22, 20, 23, 26, 24, 22, 25, 28, 26, 25]
    lows = [h - 1 for h in highs]
    lows[2], lows[6], lows[10], lows[14] = 10, 14, 12, 16
    two = add_swings(bars(highs, lows), window=2, pairs=2).iloc[-1]
    three = add_swings(bars(highs, lows), window=2, pairs=3).iloc[-1]
    assert two["hh_hl_structure"] == "uptrend"
    assert three["hh_hl_structure"] == "unclear"


def test_short_history_is_na_without_error() -> None:
    out = add_swings(ohlcv(5))
    assert out["swing_high_price"].isna().all()
    assert out["hh_hl_structure"].isna().all()
    assert out["swing_low_higher_than_prior"].isna().all()


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        add_swings(ohlcv(30), window=0)
    with pytest.raises(ValueError):
        add_swings(ohlcv(30), pairs=1)


@pytest.mark.parametrize("window", [2, 3, 5])
def test_values_do_not_depend_on_future_rows(window: int) -> None:
    # 任意の日で切っても、その日までの値が変わらない（未確定のスイングを参照していない）
    df = ohlcv(250, seed=6)
    full = add_swings(df, window=window)
    for cut in range(20, 250, 7):
        head = add_swings(df.iloc[:cut], window=window)
        pd.testing.assert_frame_equal(head, full.iloc[:cut], obj=f"cut={cut}")


def test_swing_price_appears_exactly_window_days_after_pivot() -> None:
    df = ohlcv(250, seed=7)
    out = add_swings(df, window=3)
    pivots = np.flatnonzero(pivot_highs(df["high"], 3).to_numpy())
    for i in pivots:
        if i + 3 < len(df):
            assert out["swing_high_price"].iloc[i + 3] == df["high"].iloc[i]
            assert out["swing_high_price"].iloc[i + 2] != df["high"].iloc[i]
