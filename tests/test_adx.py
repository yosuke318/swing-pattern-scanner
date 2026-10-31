import math

import numpy as np
import pandas as pd
from test_features import assert_series, ohlcv

from swing_scanner.features.adx import ADX_PERIOD, add_adx, directional_movement

nan = math.nan


def wilder_reference(df: pd.DataFrame, n: int = ADX_PERIOD) -> pd.DataFrame:
    """Wilder の原著の手順（14日合計を「前回 − 前回/14 + 当日」で更新）で書いた比較用の実装。"""
    h, lo, c = df["high"].tolist(), df["low"].tolist(), df["close"].tolist()
    tr, pdm, mdm = [], [], []
    for i in range(1, len(h)):
        tr.append(max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])))
        up, down = h[i] - h[i - 1], lo[i - 1] - lo[i]
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
    rows = [(nan, nan, nan)] * n  # 先頭行と、最初の14日合計がそろうまで
    tr14, p14, m14 = sum(tr[:n]), sum(pdm[:n]), sum(mdm[:n])
    dxs: list[float] = []
    adx = nan
    for j in range(n - 1, len(tr)):
        if j >= n:
            tr14 = tr14 - tr14 / n + tr[j]
            p14 = p14 - p14 / n + pdm[j]
            m14 = m14 - m14 / n + mdm[j]
        pdi, mdi = 100 * p14 / tr14, 100 * m14 / tr14
        dxs.append(100 * abs(pdi - mdi) / (pdi + mdi))
        if len(dxs) == n:
            adx = sum(dxs) / n
        elif len(dxs) > n:
            adx = (adx * (n - 1) + dxs[-1]) / n
        rows.append((pdi, mdi, adx))
    return pd.DataFrame(rows, columns=["plus_di", "minus_di", "adx14"])


def trend(n: int, drift: float, seed: int = 0, wave: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 1000 * np.exp(drift * t + rng.normal(0, 0.005, n))
    close *= 1 + wave * np.sin(2 * np.pi * t / 10)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_directional_movement_hand_calculated() -> None:
    high = pd.Series([10.0, 11, 12, 11.5, 13, 13.5, 14])
    low = pd.Series([9.0, 9.5, 9, 10, 10.5, 9.5, 9])
    plus, minus = directional_movement(high, low)
    # up:   -, 1, 1, -0.5, 1.5, 0.5, 0.5
    # down: -, -0.5, 0.5, -1, -0.5, 1, 0.5（最後は up = down なので両方 0）
    assert_series(plus, [nan, 1, 1, 0, 1.5, 0, 0])
    assert_series(minus, [nan, 0, 0, 0, 0, 1, 0])


def test_matches_wilder_original_procedure() -> None:
    df = ohlcv(300, seed=3)
    out = add_adx(df)
    ref = wilder_reference(df)
    for col in ["plus_di", "minus_di", "adx14"]:
        np.testing.assert_allclose(out[col], ref[col], rtol=1e-10, err_msg=col)


def test_first_valid_rows() -> None:
    out = add_adx(ohlcv(100))
    assert out["plus_di"].first_valid_index() == ADX_PERIOD
    assert out["minus_di"].first_valid_index() == ADX_PERIOD
    assert out["adx14"].first_valid_index() == 2 * ADX_PERIOD - 1


def test_uptrend_has_plus_di_above_minus_di() -> None:
    last = add_adx(trend(120, drift=0.01)).iloc[-1]
    assert last["plus_di"] > last["minus_di"]
    assert last["adx14"] > 40


def test_downtrend_has_minus_di_above_plus_di() -> None:
    last = add_adx(trend(120, drift=-0.01)).iloc[-1]
    assert last["minus_di"] > last["plus_di"]
    assert last["adx14"] > 40


def test_range_market_has_low_adx() -> None:
    assert add_adx(trend(120, drift=0.0, wave=0.03)).iloc[-1]["adx14"] < 20


def test_values_stay_in_range() -> None:
    out = add_adx(ohlcv(300, seed=4)).dropna()
    for col in ["plus_di", "minus_di", "adx14"]:
        assert ((out[col] >= 0) & (out[col] <= 100)).all(), col


def test_flat_prices_give_zero_not_error() -> None:
    df = pd.DataFrame({"high": [100.0] * 40, "low": [100.0] * 40, "close": [100.0] * 40})
    out = add_adx(df)
    assert (out[["plus_di", "minus_di"]].iloc[ADX_PERIOD:] == 0).all().all()
    assert (out["adx14"].iloc[2 * ADX_PERIOD - 1 :] == 0).all()


def test_short_history_gives_nan_without_error() -> None:
    out = add_adx(ohlcv(20))
    assert out["adx14"].isna().all()
    assert out["plus_di"].notna().iloc[ADX_PERIOD:].all()


def test_values_do_not_depend_on_future_rows() -> None:
    df = ohlcv(300, seed=5)
    full, head = add_adx(df), add_adx(df.iloc[:200])
    pd.testing.assert_frame_equal(head, full.iloc[:200])
