import csv
import datetime as dt
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from swing_scanner.config import Config, Event, load_config
from swing_scanner.daily import build_result_row
from swing_scanner.filters import (
    FilterOutcome,
    apply_filters,
    business_day_offset,
    earnings_reason,
    liquidity_reason,
    market_condition,
    missing_earnings_warnings,
)
from swing_scanner.rules.engine import evaluate
from swing_scanner.state import upsert_results

ROOT = Path(__file__).resolve().parents[1]
D = dt.date
# 2026-11-02（月）。max_holding_days = 10
TODAY = D(2026, 11, 2)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    return tmp_path / "config"


def make_config(config_dir: Path, events: list[dict] | None = None, **settings) -> Config:
    if events is not None:
        (config_dir / "events.yaml").write_text(yaml.safe_dump(events), encoding="utf-8")
    path = config_dir / "settings.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section, values in settings.items():
        data[section].update(values)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(config_dir)


def earnings(symbol: str, date: dt.date, market: str = "JP") -> dict:
    return {"symbol": symbol, "market": market, "date": date.isoformat(), "type": "earnings"}


def good_row(**overrides) -> dict:
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
        "trading_value_20d": 5e9,
    }
    return base | overrides


# --- 営業日 ------------------------------------------------------------------------


def test_business_day_offset_with_sessions() -> None:
    # 11/3（火）は文化の日
    sessions = [D(2026, 11, 2), D(2026, 11, 4), D(2026, 11, 5), D(2026, 11, 6), D(2026, 11, 9)]
    assert business_day_offset(TODAY, 1, sessions) == D(2026, 11, 4)
    assert business_day_offset(TODAY, 4, sessions) == D(2026, 11, 9)
    with pytest.raises(ValueError):
        business_day_offset(TODAY, 5, sessions)


def test_business_day_offset_without_sessions_skips_weekends() -> None:
    assert business_day_offset(D(2026, 11, 6), 1) == D(2026, 11, 9)
    assert business_day_offset(TODAY, 10) == D(2026, 11, 16)


# --- 流動性 ------------------------------------------------------------------------


def test_liquidity() -> None:
    assert liquidity_reason(2e8, 1e8) is None
    assert liquidity_reason(5e7, 1e8) == "流動性: 20日平均売買代金 50,000,000 < 下限 100,000,000"
    assert liquidity_reason(math.nan, 1e8) == "流動性: 20日平均売買代金が評価不能"


# --- 決算またぎ --------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_date, excluded",
    [
        (D(2026, 10, 30), False),  # 判定日より前
        (D(2026, 11, 2), True),  # 判定日当日（引け後発表で翌日の寄り付きが動く）
        (D(2026, 11, 16), True),  # 10営業日目
        (D(2026, 11, 17), False),  # 11営業日目
    ],
)
def test_earnings_window(event_date, excluded) -> None:
    events = [Event(symbol="7203", market="JP", date=event_date, type="earnings")]
    reason = earnings_reason(events, "JP", "7203", TODAY, D(2026, 11, 16))
    assert (reason is not None) is excluded


def test_earnings_ignores_other_symbols_markets_and_types() -> None:
    events = [
        Event(symbol="6758", market="JP", date=TODAY, type="earnings"),
        Event(symbol="7203", market="JP", date=TODAY, type="other"),
    ]
    assert earnings_reason(events, "JP", "7203", TODAY, D(2026, 11, 16)) is None


def test_earnings_window_follows_max_holding_days(config_dir: Path) -> None:
    config = make_config(config_dir, [earnings("7203", D(2026, 11, 12))])
    pattern = config.pattern("golden_cross_uptrend", 1)
    short = pattern.model_copy(
        update={"exit": pattern.exit.model_copy(update={"max_holding_days": 3})}
    )
    assert apply_filters(config, "JP", "7203", TODAY, pattern, good_row(), "uptrend").is_excluded
    assert not apply_filters(config, "JP", "7203", TODAY, short, good_row(), "uptrend").is_excluded


def test_earnings_window_uses_trading_calendar(config_dir: Path) -> None:
    # 祝日が多い週は、カレンダーで数えると窓が先まで延びる
    config = make_config(config_dir, [earnings("7203", D(2026, 11, 17))])
    pattern = config.pattern("golden_cross_uptrend", 1)
    weekdays = pd.bdate_range("2026-11-02", "2026-11-30").date
    sessions = [d for d in weekdays if d != D(2026, 11, 3)]
    assert not apply_filters(
        config, "JP", "7203", TODAY, pattern, good_row(), "uptrend"
    ).is_excluded
    outcome = apply_filters(config, "JP", "7203", TODAY, pattern, good_row(), "uptrend", sessions)
    assert outcome.excluded == ("決算: 2026-11-17 に決算（保有想定期間 2026-11-02〜2026-11-17）",)


# --- 市場環境 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "trend, mode, expected",
    [
        ("downtrend", "annotate", (None, "市場環境: 指数が downtrend")),
        ("downtrend", "suppress", ("市場環境: 指数が downtrend", None)),
        ("downtrend", "off", (None, None)),
        ("uptrend", "suppress", (None, None)),
        (pd.NA, "suppress", (None, "市場環境: 指数の trend_regime が判定不能")),
        (None, "off", (None, None)),
    ],
)
def test_market_condition(trend, mode, expected) -> None:
    assert market_condition(trend, mode) == expected


# --- まとめて当てる・個別の ON/OFF ---------------------------------------------------


def test_all_filters_combined(config_dir: Path) -> None:
    config = make_config(
        config_dir, [earnings("7203", D(2026, 11, 5))], market_filter={"mode": "suppress"}
    )
    pattern = config.pattern("golden_cross_uptrend", 1)
    outcome = apply_filters(
        config, "JP", "7203", TODAY, pattern, good_row(trading_value_20d=1e7), "downtrend"
    )
    assert [r.split(":")[0] for r in outcome.excluded] == ["流動性", "決算", "市場環境"]


def test_each_filter_can_be_turned_off(config_dir: Path) -> None:
    config = make_config(
        config_dir,
        [earnings("7203", D(2026, 11, 5))],
        filters={"liquidity": False, "earnings": False},
        market_filter={"mode": "off"},
    )
    pattern = config.pattern("golden_cross_uptrend", 1)
    outcome = apply_filters(
        config, "JP", "7203", TODAY, pattern, good_row(trading_value_20d=1e7), "downtrend"
    )
    assert outcome == FilterOutcome((), ())


def test_liquidity_minimum_is_per_market(config_dir: Path) -> None:
    config = make_config(config_dir, [])
    pattern = config.pattern("golden_cross_uptrend", 1)
    row = good_row(trading_value_20d=5e7)  # 日本株の下限 1億円未満、米国株の下限 1,000万ドル以上
    assert apply_filters(config, "JP", "7203", TODAY, pattern, row, "uptrend").is_excluded
    assert not apply_filters(config, "US", "NVDA", TODAY, pattern, row, "uptrend").is_excluded


# --- 決算日の登録漏れ ----------------------------------------------------------------


def test_missing_earnings_warnings(config_dir: Path) -> None:
    config = make_config(
        config_dir,
        [
            earnings("7203", D(2026, 8, 3)),  # 91日前 → 警告
            earnings("6758", D(2026, 8, 4)),  # ちょうど90日前 → 警告しない
            earnings("8035", D(2026, 11, 20)),  # 未来の登録 → 警告しない
        ],
    )
    warnings = missing_earnings_warnings(config, "JP", TODAY)
    warned = {w.split(" ")[0] for w in warnings}
    jp = {s.symbol for s in config.watched_symbols("JP")}
    assert warned == jp - {"6758", "8035"}
    assert "7203" in warned
    assert all("events.yaml に登録されていません" in w for w in warnings)


def test_no_registration_warning_when_earnings_filter_off(config_dir: Path) -> None:
    config = make_config(config_dir, [], filters={"earnings": False})
    assert missing_earnings_warnings(config, "JP", TODAY) == []


# --- 判定ログ（[3-3]） --------------------------------------------------------------


def test_result_rows_record_everything(config_dir: Path, tmp_path: Path) -> None:
    config = make_config(config_dir, [earnings("6758", D(2026, 11, 5))])
    pattern = config.pattern("golden_cross_uptrend", 1)
    at = dt.datetime(2026, 11, 2, 9, 0, tzinfo=dt.UTC)
    cases = {
        "7203": good_row(),  # 通知
        "6758": good_row(),  # 決算で除外
        "8035": good_row(adx14=15.0),  # 惜しい（75点）
        "9983": good_row(vol_regime="high", trading_value_20d=1e7),  # レジーム対象外 + 流動性
    }
    rows = []
    for symbol, r in cases.items():
        ev = evaluate(pattern, r)
        outcome = apply_filters(config, "JP", symbol, TODAY, pattern, r, "uptrend")
        notified = ev.matched and not outcome.is_excluded
        rows.append(
            build_result_row(TODAY, "JP", symbol, pattern, ev, outcome, notified, "abc123", at)
        )
    upsert_results(tmp_path, rows)
    upsert_results(tmp_path, rows)  # 同じ対象日の再実行で二重にならない

    with (tmp_path / "results/2026-11.csv").open(encoding="utf-8", newline="") as f:
        log = {r["symbol"]: r for r in csv.DictReader(f)}
    assert len(log) == 4
    assert (log["7203"]["matched"], log["7203"]["notified"], log["7203"]["excluded_reason"]) == (
        "true",
        "true",
        "",
    )
    assert (log["7203"]["stop"], log["7203"]["target"]) == ("980.0", "1060.0")
    assert log["6758"]["notified"] == "false"
    assert log["6758"]["excluded_reason"].startswith("決算: 2026-11-05")
    assert (log["8035"]["score"], log["8035"]["matched"]) == ("75.0", "false")
    breakdown = json.loads(log["8035"]["breakdown"])
    assert {c["key"]: c["passed"] for c in breakdown}["adx14"] is False
    assert {c["key"]: c["actual"] for c in breakdown}["adx14"] == 15.0
    reasons = log["9983"]["excluded_reason"].split("; ")
    assert reasons[0] == "レジーム対象外: vol_regime=high（対象: normal）"
    assert reasons[1].startswith("流動性")
    assert {r["commit_sha"] for r in log.values()} == {"abc123"}
    assert {(r["pattern_id"], r["version"]) for r in log.values()} == {
        ("golden_cross_uptrend", "1")
    }


def test_result_row_without_exit(config_dir: Path) -> None:
    config = make_config(config_dir, [])
    pattern = config.pattern("golden_cross_uptrend", 1)
    ev = evaluate(pattern, good_row(atr14=math.nan))
    row = build_result_row(
        TODAY,
        "JP",
        "7203",
        pattern,
        ev,
        FilterOutcome((), ()),
        False,
        "abc",
        dt.datetime.now(dt.UTC),
    )
    assert (row.stop, row.target, row.max_holding_days) == (None, None, 10)
