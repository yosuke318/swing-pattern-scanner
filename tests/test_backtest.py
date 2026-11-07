"""バックテスト（[4-1] 基盤、[4-2] 仮想トレード）。"""

import datetime as dt
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_features import ohlcv

from swing_scanner.backtest.cache import OhlcvCache
from swing_scanner.backtest.report import render_markdown, summarize, write_report
from swing_scanner.backtest.runner import run_backtest
from swing_scanner.backtest.trades import Trade, simulate_trade
from swing_scanner.cli import main
from swing_scanner.config import ExitRule, load_config
from swing_scanner.features import compute_features
from swing_scanner.rules.engine import evaluate

ROOT = Path(__file__).resolve().parents[1]
D = dt.date
JP = ["7203", "6758", "8306", "8035", "9984"]


def rule(stop_type="atr", k=1.0, m=2.0, days=3, gap=None) -> ExitRule:
    data = {
        "stop": {"type": stop_type, "atr_mult": k},
        "target": {"atr_mult": m},
        "max_holding_days": days,
    }
    if gap is not None:
        data["skip_gap_atr_mult"] = gap
    return ExitRule.model_validate(data)


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """(始値, 高値, 安値, 終値) の並び。0行目がシグナル日。"""
    dates = pd.bdate_range("2026-01-05", periods=len(rows)).date
    return pd.DataFrame(
        [
            {"date": d, "open": o, "high": h, "low": lo, "close": c}
            for d, (o, h, lo, c) in zip(dates, rows, strict=True)
        ]
    )


def trade(rows, r=None, atr=10.0, swing=None, cost=0.0) -> Trade | str:
    return simulate_trade(bars(rows), 0, r or rule(), atr, swing, cost, "7203", "p", 1)


SIGNAL = (100, 101, 99, 100)


# --- 仮想トレード（[4-2]） ----------------------------------------------------------


def test_entry_is_next_open_and_target_hit() -> None:
    # 翌日寄り付き 102 で入り、損切り 92・利確 122。3日目の高値 125 で利確
    t = trade([SIGNAL, (102, 110, 98, 108), (108, 125, 105, 120)])
    assert (t.entry_date, t.entry_price, t.stop, t.target) == (D(2026, 1, 6), 102, 92, 122)
    assert (t.exit_reason, t.exit_price, t.exit_date, t.holding_days) == (
        "target",
        122,
        D(2026, 1, 7),
        1,
    )
    assert t.gross_return == pytest.approx(122 / 102 - 1)


def test_stop_hit_on_entry_day() -> None:
    t = trade([SIGNAL, (102, 103, 90, 95)])
    assert (t.exit_reason, t.exit_price, t.holding_days) == ("stop", 92, 0)


def test_same_day_stop_and_target_counts_stop_first() -> None:
    t = trade([SIGNAL, (102, 103, 100, 101), (101, 130, 85, 110)])
    assert (t.exit_reason, t.exit_price) == ("stop", 92)


def test_gap_down_exits_at_open() -> None:
    t = trade([SIGNAL, (102, 105, 99, 100), (80, 85, 75, 82)])
    assert (t.exit_reason, t.exit_price) == ("stop", 80)


def test_gap_up_above_target_exits_at_open() -> None:
    t = trade([SIGNAL, (102, 105, 99, 100), (130, 135, 128, 131)])
    assert (t.exit_reason, t.exit_price) == ("target", 130)


def test_time_exit_at_next_open() -> None:
    rows = [
        SIGNAL,
        (102, 105, 99, 103),
        (103, 106, 100, 104),
        (104, 107, 101, 105),
        (106, 108, 103, 107),
    ]
    t = trade(rows, rule(days=3))
    assert (t.exit_reason, t.exit_price, t.exit_date, t.holding_days) == (
        "time",
        106,
        D(2026, 1, 9),
        3,
    )
    assert t.mfe == pytest.approx(107 / 102 - 1)
    assert t.mae == pytest.approx(99 / 102 - 1)


def test_end_of_data_closes_at_last_close() -> None:
    t = trade([SIGNAL, (102, 105, 99, 103), (103, 106, 100, 104)], rule(days=10))
    assert (t.exit_reason, t.exit_price) == ("end_of_data", 104)


def test_gap_up_entry_is_skipped() -> None:
    # シグナル日の終値 100 + 1×ATR10 = 110 以上で寄り付いたら見送る
    assert trade([SIGNAL, (110, 112, 108, 111)], rule(gap=1.0)) == "gap"
    assert isinstance(trade([SIGNAL, (109, 112, 108, 111)], rule(gap=1.0)), Trade)


def test_signal_on_last_bar_has_no_trade() -> None:
    assert trade([SIGNAL]) == "no_next_bar"
    assert trade([SIGNAL, (102, 105, 99, 103)], atr=math.nan) == "no_exit_plan"


def test_stop_is_recomputed_from_entry_price() -> None:
    # スイング安値 95 は エントリー 102 − 1×ATR = 92 より浅いので採用。エントリーより上なら使わない
    t = trade([SIGNAL, (102, 103, 94, 95)], rule("swing_low_or_atr"), swing=95)
    assert (t.stop, t.stop_basis, t.exit_reason, t.exit_price) == (95, "swing_low", "stop", 95)
    above = trade([SIGNAL, (102, 103, 94, 95)], rule("swing_low_or_atr"), swing=103)
    assert (above.stop, above.stop_basis) == (92, "atr")


def test_cost_is_deducted_both_ways() -> None:
    t = trade([SIGNAL, (100, 110, 99, 105), (105, 125, 104, 120)], cost=0.001)
    assert t.net_return == pytest.approx(120 * 0.999 / (100 * 1.001) - 1)
    assert t.net_return < t.gross_return


def test_trade_uses_only_bars_after_signal() -> None:
    rows = [
        (90, 91, 89, 90),
        (95, 96, 94, 95),
        SIGNAL,
        (102, 110, 98, 108),
        (108, 125, 105, 120),
        (50, 60, 40, 55),
    ]
    base = simulate_trade(bars(rows), 2, rule(), 10.0, None, 0.0, "7203", "p", 1)
    # シグナル日より前の足や、決済より後の足を変えても結果は変わらない
    changed = [(10, 11, 9, 10), (500, 600, 400, 500), *rows[2:5], (999, 999, 1, 5)]
    assert simulate_trade(bars(changed), 2, rule(), 10.0, None, 0.0, "7203", "p", 1) == base


# --- キャッシュ ---------------------------------------------------------------------


class CountingSource:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, dt.date, dt.date]] = []

    def fetch(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if symbol not in self.frames:
            raise FileNotFoundError(symbol)
        df = self.frames[symbol]
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


LAST_DAY = D(2026, 10, 30)


def frame(seed: int, end: dt.date = LAST_DAY, n: int = 900) -> pd.DataFrame:
    df = ohlcv(n, seed=seed)
    df["date"] = pd.bdate_range(end=end, periods=n).date
    return df


def test_cache_reads_without_calling_source(tmp_path: Path) -> None:
    src = CountingSource({"7203": frame(1)})
    t0 = dt.datetime(2026, 11, 1, tzinfo=dt.UTC)
    cache = OhlcvCache(src, tmp_path, now=t0)
    first = cache.load("JP", "7203", D(2024, 1, 1), D(2026, 10, 30))
    assert not first.from_cache and len(src.calls) == 1
    second = OhlcvCache(src, tmp_path).load("JP", "7203", D(2024, 1, 1), D(2026, 10, 30))
    assert second.from_cache and len(src.calls) == 1
    assert second.fetched_at == t0
    pd.testing.assert_frame_equal(second.data, first.data)
    # 範囲の内側ならキャッシュから切り出す
    inner = OhlcvCache(src, tmp_path).load("JP", "7203", D(2025, 1, 1), D(2025, 12, 31))
    assert inner.from_cache and inner.data["date"].min() >= D(2025, 1, 1)


def test_cache_refetches_when_range_grows_or_refresh(tmp_path: Path) -> None:
    src = CountingSource({"7203": frame(1)})
    cache = OhlcvCache(src, tmp_path)
    cache.load("JP", "7203", D(2025, 1, 1), D(2026, 10, 30))
    cache.load("JP", "7203", D(2024, 1, 1), D(2026, 10, 30))  # 前に延びた
    cache.load("JP", "7203", D(2024, 1, 1), D(2026, 10, 30), refresh=True)
    assert len(src.calls) == 3


# --- 実行（[4-1]） ------------------------------------------------------------------


ALWAYS = {
    "pattern_id": "always",
    "version": 1,
    "name": "常に当たる",
    "category": "trend",
    "active": True,
    "regimes": {"vol": ["low", "normal", "high"], "trend": ["uptrend", "range", "downtrend"]},
    "conditions": [{"key": "close", "op": ">", "value": 0, "weight": 100}],
    "score_threshold": 100,
    "exit": {
        "stop": {"type": "atr", "atr_mult": 1.5},
        "target": {"atr_mult": 3},
        "max_holding_days": 5,
    },
}


@pytest.fixture
def config(tmp_path: Path):
    shutil.copytree(ROOT / "config", tmp_path / "config")
    (tmp_path / "config/patterns/always").mkdir()
    (tmp_path / "config/patterns/always/v1.yaml").write_text(
        yaml.safe_dump(ALWAYS), encoding="utf-8"
    )
    return load_config(tmp_path / "config")


def sources() -> CountingSource:
    return CountingSource({code: frame(i) for i, code in enumerate(JP)} | {"1306": frame(99)})


def run(config, tmp_path, pattern_id="always", src=None, **kw):
    symbols = config.watched_symbols("JP")
    cache = OhlcvCache(src or sources(), tmp_path / "cache")
    pattern = config.pattern(pattern_id)
    return run_backtest(config, pattern, "JP", symbols, D(2025, 1, 1), D(2026, 10, 30), cache, **kw)


def test_trades_do_not_overlap_and_follow_cooldown(config, tmp_path: Path) -> None:
    result = run(config, tmp_path)
    assert result.trades
    by_symbol: dict[str, list[Trade]] = {}
    for t in result.trades:
        by_symbol.setdefault(t.symbol, []).append(t)
        assert t.entry_date > t.signal_date  # シグナル日の終値では入らない
        assert D(2025, 1, 1) <= t.signal_date <= D(2026, 10, 30)
    for trades in by_symbol.values():
        for prev, nxt in zip(trades, trades[1:], strict=False):
            assert nxt.signal_date >= prev.exit_date  # 保有中は新規に入らない
            assert (nxt.signal_date - prev.signal_date).days >= 5  # クールダウン（5営業日）
    assert result.cooled_down > 0
    assert {s.skip_reason for s in result.signals} <= {"", "holding", "no_next_bar"}


def test_signals_match_daily_judgement(config, tmp_path: Path) -> None:
    # バックテストのシグナル日に、日次処理と同じく約300営業日だけを入力して判定しても通知対象になる
    src = sources()
    pattern = config.pattern("golden_cross_uptrend")
    result = run(config, tmp_path, "golden_cross_uptrend", src)
    bench = src.frames["1306"]
    lookback = config.settings.fetch.lookback_days
    for s in result.signals[:10]:
        df = src.frames[s.symbol]
        window = df[df["date"] <= s.date].tail(lookback)
        bwin = bench[bench["date"] <= s.date]
        row = compute_features(window, bwin, config.settings.features).iloc[-1]
        ev = evaluate(pattern, row)
        assert not ev.skip_reason and ev.score >= config.score_threshold(pattern), s


def test_liquidity_exclusions_are_counted(config, tmp_path: Path) -> None:
    data = config.settings.model_dump()
    data["markets"]["JP"]["min_trading_value_20d"] = 1e15
    strict = config.model_copy(update={"settings": type(config.settings).model_validate(data)})
    result = run(strict, tmp_path)
    assert result.trades == []
    assert result.excluded.get("流動性", 0) > 0


def test_missing_symbol_is_reported_and_others_continue(config, tmp_path: Path) -> None:
    src = sources()
    del src.frames["6758"]
    result = run(config, tmp_path, src=src)
    errors = {d.symbol: d.error for d in result.data if d.error}
    assert list(errors) == ["6758"]
    assert {t.symbol for t in result.trades} <= set(JP) - {"6758"}


# --- レポート ----------------------------------------------------------------------


def test_report_records_conditions_and_is_reproducible(config, tmp_path: Path) -> None:
    src = sources()
    first = run(config, tmp_path, src=src)
    calls = len(src.calls)
    second = run(config, tmp_path, src=src)  # 2回目はキャッシュから読む
    assert len(src.calls) == calls
    assert first.trades == second.trades

    at = dt.datetime(2026, 11, 7, 1, 0, tzinfo=dt.UTC)
    md, csv_path = write_report(
        first, config, "swing-scanner backtest ...", "abc123", at, tmp_path / "out"
    )
    assert md.name == "20261107-010000_always_v1.md"
    text = md.read_text(encoding="utf-8")
    for expected in [
        "コミット: `abc123`",
        "コマンド: `swing-scanner backtest ...`",
        "期間: 2025-01-01 〜 2026-10-30",
        "想定コスト（片道）: 0.1000%",
        "pattern_id: always",
        "golden_cross_pair:",
        "| 1306（ベンチマーク） |",
        "生存バイアス",
    ]:
        assert expected in text, expected
    rows = pd.read_csv(csv_path)
    assert len(rows) == len(first.trades)


def test_summary_metrics() -> None:
    def t(net: float, reason: str = "stop") -> Trade:
        return Trade(
            "7203",
            "p",
            1,
            D(2026, 1, 5),
            D(2026, 1, 6),
            100,
            90,
            "atr",
            120,
            D(2026, 1, 8),
            100,
            reason,
            2,
            net,
            net,
            0.05,
            -0.03,
        )

    class R:
        trades = [t(0.10, "target"), t(-0.05), t(-0.05), t(0.02, "time")]
        signals = []

    s = summarize(R())
    assert (s.trades, s.wins, s.win_rate) == (4, 2, 0.5)
    assert s.avg_net_return == pytest.approx(0.005)
    assert s.profit_factor == pytest.approx(0.12 / 0.10)
    assert s.exit_reasons == {"target": 1, "stop": 2, "time": 1}


def test_empty_result_renders(config, tmp_path: Path) -> None:
    result = run(config, tmp_path, "squeeze_breakout")
    text = render_markdown(result, config, "cmd", "abc", dt.datetime(2026, 11, 7, tzinfo=dt.UTC))
    assert "# バックテスト: スクイーズからのブレイク" in text


# --- CLI ---------------------------------------------------------------------------


def test_cli_backtest(tmp_path: Path, capsys) -> None:
    data = tmp_path / "csv"
    data.mkdir()
    for i, code in enumerate(JP + ["1306"]):
        frame(i if code != "1306" else 99).to_csv(data / f"{code}.csv", index=False)
    args = [
        "backtest",
        "--pattern",
        "ma_pullback",
        "--market",
        "JP",
        "--start",
        "2025-01-01",
        "--end",
        "2026-10-30",
        "--data-dir",
        str(data),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "レポート:" in out
    assert len(list((tmp_path / "out").glob("*_ma_pullback_v1.md"))) == 1
    assert main([*args[:2], "no_such", *args[3:]]) == 1
    assert (
        main(
            [
                "backtest",
                "--pattern",
                "ma_pullback",
                "--market",
                "JP",
                "--start",
                "2026-01-01",
                "--end",
                "2025-01-01",
            ]
        )
        == 1
    )
