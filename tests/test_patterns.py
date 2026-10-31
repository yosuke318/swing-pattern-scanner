"""config/patterns/ の初期パターン（[3-7]）の性質。"""

import datetime as dt
import math
from pathlib import Path

import pytest

from swing_scanner.config import load_config
from swing_scanner.daily import build_result_row
from swing_scanner.filters import FilterOutcome
from swing_scanner.rules.engine import evaluate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config")
ACTIVE = CONFIG.active_patterns()


def test_four_active_patterns() -> None:
    assert sorted(p.pattern_id for p in ACTIVE) == [
        "golden_cross_uptrend",
        "ma_pullback",
        "squeeze_breakout",
        "volume_breakout",
    ]


@pytest.mark.parametrize("pattern", ACTIVE, ids=lambda p: p.pattern_id)
def test_only_uptrend_and_requires_relative_strength(pattern) -> None:
    assert pattern.regimes.trend == ["uptrend"]
    required = {c.key: c for c in pattern.conditions if c.required}
    assert "relative_strength_20d" in required
    assert required["relative_strength_20d"].op == ">="
    assert 5 <= pattern.exit.max_holding_days <= 20


def test_regimes_follow_design() -> None:
    vol = {p.pattern_id: set(p.regimes.vol) for p in ACTIVE}
    assert vol == {
        "golden_cross_uptrend": {"normal"},
        "ma_pullback": {"low", "normal"},
        "squeeze_breakout": {"low"},
        "volume_breakout": {"high"},
    }


def strong_row(**overrides) -> dict:
    """どのパターンの条件も満たす1日分の指標。"""
    base = {
        "close": 1050.0,
        "atr14": 20.0,
        "swing_low_price": 1020.0,
        "swing_high_price": 1040.0,
        "days_since_golden_cross": 2.0,
        "swing_low_higher_than_prior": True,
        "hh_hl_structure": "uptrend",
        "adx14": 30.0,
        "plus_di": 30.0,
        "minus_di": 15.0,
        "close_minus_ma25_atr_ratio": 0.3,
        "volume_ratio_20d": 2.5,
        "relative_strength_20d": 0.02,
        "trend_regime": "uptrend",
    }
    return base | overrides


@pytest.mark.parametrize("vol", ["low", "normal", "high"])
def test_each_vol_regime_gets_only_its_patterns(vol: str) -> None:
    applied = {p.pattern_id for p in ACTIVE if evaluate(p, strong_row(vol_regime=vol)).matched}
    expected = {p.pattern_id for p in ACTIVE if vol in p.regimes.vol}
    assert applied == expected
    assert applied  # どのボラでも当たるパターンが1つはある


@pytest.mark.parametrize("trend", ["downtrend", "range"])
def test_no_buy_signal_outside_uptrend(trend: str) -> None:
    for vol in ("low", "normal", "high"):
        for p in ACTIVE:
            ev = evaluate(p, strong_row(vol_regime=vol, trend_regime=trend))
            assert not ev.matched
            if vol in p.regimes.vol:
                assert ev.skip_reason == f"trend_regime={trend}（対象: uptrend）"


@pytest.mark.parametrize("rs", [-0.01, math.nan])
def test_weak_or_unknown_relative_strength_blocks_signal(rs: float) -> None:
    for vol in ("low", "normal", "high"):
        for p in ACTIVE:
            if vol not in p.regimes.vol:
                continue
            ev = evaluate(p, strong_row(vol_regime=vol, relative_strength_20d=rs))
            assert not ev.matched
            assert ev.skip_reason.startswith("必須条件未達")
            assert "relative_strength_20d" in ev.skip_reason
            # スコアは他の条件の分だけ残る（判定ログで惜しさが分かる）
            assert ev.score >= 75


def test_required_failure_is_logged() -> None:
    p = CONFIG.pattern("ma_pullback")
    ev = evaluate(p, strong_row(vol_regime="normal", relative_strength_20d=-0.05))
    row = build_result_row(
        dt.date(2026, 11, 2),
        "JP",
        "7203",
        p,
        ev,
        FilterOutcome((), ()),
        False,
        "abc",
        dt.datetime.now(dt.UTC),
    )
    assert row.excluded_reason == "必須条件未達: relative_strength_20d"
    rs = next(c for c in row.breakdown if c["key"] == "relative_strength_20d")
    assert (rs["passed"], rs["required"]) == (False, True)


def test_breakout_requires_new_high() -> None:
    p = CONFIG.pattern("squeeze_breakout")
    ev = evaluate(p, strong_row(vol_regime="low", swing_high_price=1100.0))
    assert ev.skip_reason == "必須条件未達: close_above_swing_high"
