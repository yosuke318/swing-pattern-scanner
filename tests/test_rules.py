import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_features import ohlcv

from swing_scanner.config import ExitRule, Pattern, load_config
from swing_scanner.features import EXTRA_COLUMNS, FEATURE_COLUMNS, compute_features
from swing_scanner.rules.engine import compare, evaluate, evaluate_all, exit_plan
from swing_scanner.rules.keys import KEYS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config")
# golden_cross_within_days <= 5 / swing_low_higher_than_prior == true / adx14 >= 20 /
# close_minus_ma25_atr_ratio between [-1.0, 0.5]、各 25 点。vol: normal、trend: uptrend
PATTERN = CONFIG.pattern("golden_cross_uptrend", 1)


def row(**overrides) -> dict:
    base = {
        "close": 1000.0,
        "atr14": 20.0,
        "swing_low_price": 980.0,
        "days_since_golden_cross": 3.0,
        "swing_low_higher_than_prior": True,
        "adx14": 25.0,
        "close_minus_ma25_atr_ratio": 0.2,
        "vol_regime": "normal",
        "trend_regime": "uptrend",
    }
    return base | overrides


def statuses(ev) -> dict[str, str]:
    return {c.key: c.status for c in ev.conditions}


# --- 条件の評価とスコア ------------------------------------------------------------


def test_all_conditions_pass() -> None:
    ev = evaluate(PATTERN, row())
    assert (ev.score, ev.matched, ev.skip_reason) == (100, True, "")
    assert set(statuses(ev).values()) == {"passed"}


def test_partial_match_keeps_score() -> None:
    ev = evaluate(PATTERN, row(adx14=15.0))
    assert (ev.score, ev.matched) == (75, False)
    assert statuses(ev)["adx14"] == "failed"
    assert ev.conditions[2].actual == 15.0 and ev.conditions[2].threshold == 20


def test_golden_cross_never_happened_is_failed_not_unevaluable() -> None:
    ev = evaluate(PATTERN, row(days_since_golden_cross=math.inf))
    assert statuses(ev)["golden_cross_within_days"] == "failed"
    assert ev.breakdown()[0]["actual"] == "inf"


@pytest.mark.parametrize(
    "overrides, key",
    [
        ({"adx14": math.nan}, "adx14"),
        ({"adx14": None}, "adx14"),
        ({"swing_low_higher_than_prior": pd.NA}, "swing_low_higher_than_prior"),
        ({"days_since_golden_cross": np.nan}, "golden_cross_within_days"),
    ],
)
def test_missing_indicator_is_unevaluable_and_never_matches(overrides, key) -> None:
    ev = evaluate(PATTERN, row(**overrides))
    assert statuses(ev)[key] == "unevaluable"
    assert ev.unevaluable_keys == [key]
    assert (ev.score, ev.matched) == (75, False)
    assert next(c for c in ev.breakdown() if c["key"] == key)["passed"] is None


def test_missing_column_is_unevaluable() -> None:
    r = row()
    del r["adx14"]
    assert statuses(evaluate(PATTERN, r))["adx14"] == "unevaluable"


@pytest.mark.parametrize("ratio, passed", [(-1.0, True), (0.5, True), (0.5000001, False)])
def test_between_includes_both_ends(ratio, passed) -> None:
    ev = evaluate(PATTERN, row(close_minus_ma25_atr_ratio=ratio))
    assert (statuses(ev)["close_minus_ma25_atr_ratio"] == "passed") is passed


@pytest.mark.parametrize(
    "op, actual, threshold, expected",
    [
        ("<", 1, 2, True),
        ("<", 2, 2, False),
        ("<=", 2, 2, True),
        (">", 2, 2, False),
        (">=", 2, 2, True),
        ("==", True, True, True),
        ("!=", True, True, False),
        ("between", 3, [1, 3], True),
    ],
)
def test_compare(op, actual, threshold, expected) -> None:
    assert compare(op, actual, threshold) is expected


def test_numpy_and_pandas_values_are_accepted() -> None:
    ev = evaluate(
        PATTERN, pd.Series(row(adx14=np.float64(25), swing_low_higher_than_prior=np.True_))
    )
    assert ev.matched
    json.dumps(ev.breakdown(), allow_nan=False)


# --- レジーム ----------------------------------------------------------------------


def test_regime_mismatch_is_not_applied() -> None:
    ev = evaluate(PATTERN, row(vol_regime="high"))
    assert (ev.score, ev.matched) == (100, False)
    assert ev.skip_reason == "vol_regime=high（対象: normal）"


def test_unevaluable_regime_is_not_applied() -> None:
    ev = evaluate(PATTERN, row(trend_regime=pd.NA))
    assert not ev.matched
    assert ev.skip_reason == "trend_regime が評価不能"


# --- 出口 --------------------------------------------------------------------------


def test_exit_uses_shallower_of_swing_low_and_atr() -> None:
    # ATR 側 1000 − 1.5×20 = 970、スイング安値 980 の方が浅い
    plan = evaluate(PATTERN, row()).exit
    assert (plan.stop, plan.stop_basis, plan.target, plan.max_holding_days) == (
        980,
        "swing_low",
        1060,
        10,
    )
    assert plan.skip_if_open_at_or_above is None


@pytest.mark.parametrize("swing", [960.0, 1005.0, math.nan, None])
def test_exit_falls_back_to_atr(swing) -> None:
    # 深すぎる / すでに割り込んでいる / 無い スイング安値は使わない
    plan = exit_plan(PATTERN.exit, 1000.0, 20.0, swing)
    assert (plan.stop, plan.stop_basis) == (970, "atr")


def test_exit_atr_type_and_gap() -> None:
    rule = ExitRule.model_validate(
        {
            "stop": {"type": "atr", "atr_mult": 2},
            "target": {"atr_mult": 4},
            "max_holding_days": 5,
            "skip_gap_atr_mult": 1.0,
        }
    )
    plan = exit_plan(rule, 1000.0, 20.0, 990.0)
    assert (plan.stop, plan.stop_basis, plan.target) == (960, "atr", 1080)
    assert plan.skip_if_open_at_or_above == 1020


def test_exit_can_be_recomputed_from_entry_price() -> None:
    # バックテストは翌日寄り付き（エントリー価格）を基準に計算し直す
    plan = exit_plan(PATTERN.exit, 1010.0, 20.0, 980.0)
    assert (plan.base_price, plan.stop, plan.target) == (1010, 980, 1070)


def test_exit_is_none_without_atr() -> None:
    assert evaluate(PATTERN, row(atr14=math.nan)).exit is None


# --- key の一覧 --------------------------------------------------------------------


def test_derived_keys() -> None:
    r = {"plus_di": 30.0, "minus_di": 20.0, "hh_hl_structure": "unclear"}
    assert KEYS["plus_di_above_minus_di"].value(r) is True
    assert KEYS["hh_hl_uptrend"].value(r) is False
    assert KEYS["hh_hl_uptrend"].value({"hh_hl_structure": pd.NA}) is None
    swing = {"swing_high_price": 110.0, "prior_swing_high_price": 100.0}
    assert KEYS["swing_high_higher_than_prior"].value(swing) is True


def test_every_key_reads_existing_feature_columns() -> None:
    columns = set(FEATURE_COLUMNS + EXTRA_COLUMNS)
    for spec in KEYS.values():
        assert set(spec.columns) <= columns, spec.name


def test_every_key_is_documented_in_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### パターン条件で使える key", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^\| `([a-z0-9_]+)` \|", section, flags=re.M))
    assert documented == set(KEYS)


def test_unknown_key_cannot_be_defined() -> None:
    data = PATTERN.model_dump()
    data["conditions"][0]["key"] = "rsi14"
    with pytest.raises(ValueError, match="未知の key"):
        Pattern.model_validate(data)


# --- 指標の DataFrame と通しで -------------------------------------------------------


def test_evaluate_rows_from_compute_features() -> None:
    features = compute_features(ohlcv(600, seed=13), ohlcv(600, seed=14), CONFIG.settings.features)
    for _, r in features.iloc[300:].iterrows():
        for ev in evaluate_all(CONFIG.active_patterns(), r):
            assert 0 <= ev.score <= 100
            assert not ev.unevaluable_keys
            json.dumps(ev.breakdown(), allow_nan=False)
            assert ev.exit is not None and ev.exit.stop < ev.exit.base_price < ev.exit.target
