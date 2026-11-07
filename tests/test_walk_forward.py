"""ウォークフォワード検証（[4-3]）。"""

import datetime as dt
from pathlib import Path

import pytest
from test_backtest import JP, config, frame, sources  # noqa: F401  (config は fixture)

from swing_scanner.backtest.cache import OhlcvCache
from swing_scanner.backtest.walk_forward import (
    Params,
    achievable_scores,
    add_months,
    candidate_params,
    make_windows,
    render_walk_forward,
    run_walk_forward,
    write_walk_forward,
)
from swing_scanner.cli import main

D = dt.date


def test_add_months_clamps_to_month_end() -> None:
    assert add_months(D(2024, 1, 31), 1) == D(2024, 2, 29)
    assert add_months(D(2023, 11, 15), 14) == D(2025, 1, 15)


def test_windows_slide_and_clip_to_end() -> None:
    windows = make_windows(D(2021, 1, 1), D(2025, 9, 30), 24, 12, 12)
    assert [(w.train_start, w.train_end, w.test_start, w.test_end) for w in windows] == [
        (D(2021, 1, 1), D(2022, 12, 31), D(2023, 1, 1), D(2023, 12, 31)),
        (D(2022, 1, 1), D(2023, 12, 31), D(2024, 1, 1), D(2024, 12, 31)),
        (D(2023, 1, 1), D(2024, 12, 31), D(2025, 1, 1), D(2025, 9, 30)),
    ]
    assert make_windows(D(2024, 1, 1), D(2025, 6, 30), 24, 12, 12) == []


def test_threshold_candidates_come_from_weights(config) -> None:  # noqa: F811
    assert achievable_scores(config.pattern("ma_pullback")) == [50, 75, 100]
    assert achievable_scores(config.pattern("golden_cross_uptrend")) == [60, 80, 100]
    grid = config.settings.backtest.walk_forward
    n = len(grid.stop_atr_mult) * len(grid.target_atr_mult) * len(grid.max_holding_days)
    assert len(candidate_params(config, config.pattern("ma_pullback"))) == 3 * n


def test_params_change_only_threshold_and_exit(config) -> None:  # noqa: F811
    pattern = config.pattern("golden_cross_uptrend")
    variant = Params(60, 2.0, 4.0, 15).apply(pattern)
    assert variant.score_threshold == 60
    assert (variant.exit.stop.atr_mult, variant.exit.target.atr_mult) == (2.0, 4.0)
    assert variant.exit.max_holding_days == 15
    assert variant.exit.stop.type == pattern.exit.stop.type
    assert variant.conditions == pattern.conditions and variant.regimes == pattern.regimes


def run(config, tmp_path: Path, src, **kw):  # noqa: F811
    cache = OhlcvCache(src, tmp_path)
    return run_walk_forward(
        config,
        config.pattern("ma_pullback"),
        "JP",
        config.watched_symbols("JP"),
        D(2023, 1, 1),
        D(2026, 10, 30),
        cache,
        train_months=24,
        test_months=12,
        step_months=12,
        **kw,
    )


def test_train_and_test_are_reported_side_by_side(config, tmp_path: Path) -> None:  # noqa: F811
    result = run(config, tmp_path, sources())
    assert len(result.windows) == 2
    for w in result.windows:
        assert w.params is not None and w.train and w.test
        # 学習期間のトレードは学習期間の中で始まり、学習期間の終わりより後の足を使わない
        for t in w.train.trades:
            assert w.window.train_start <= t.signal_date <= w.window.train_end
            assert t.exit_date <= w.window.train_end
        for t in w.test.trades:
            assert w.window.test_start <= t.signal_date <= w.window.test_end
        assert w.train_summary.trades >= result.min_trades
    assert len(result.test_trades) == sum(len(w.test.trades) for w in result.windows)


def test_test_period_prices_do_not_leak_into_selection(config, tmp_path: Path) -> None:  # noqa: F811
    base = run(config, tmp_path / "a", sources())
    # 最初の窓の評価期間（2025-01-01 以降）の値動きを大きく変える
    changed = sources()
    for code, df in changed.frames.items():
        later = df["date"] >= D(2025, 1, 1)
        for col in ("open", "high", "low", "close"):
            df.loc[later, col] = df.loc[later, col] * 1.5
        changed.frames[code] = df
    other = run(config, tmp_path / "b", changed)
    first, second = base.windows[0], other.windows[0]
    assert first.params == second.params
    assert first.train.trades == second.train.trades
    assert first.test.trades != second.test.trades  # 評価期間の成績だけが変わる


def test_too_few_trades_are_flagged(config, tmp_path: Path) -> None:  # noqa: F811
    data = config.settings.model_dump()
    data["backtest"]["walk_forward"]["min_trades"] = 10_000
    strict = config.model_copy(update={"settings": type(config.settings).model_validate(data)})
    result = run(strict, tmp_path, sources())
    assert all(w.params is None and w.test is None for w in result.windows)
    text = render_walk_forward(result, "cmd", "abc", dt.datetime(2026, 11, 7, tzinfo=dt.UTC))
    assert "選べない" in text


def test_report(config, tmp_path: Path) -> None:  # noqa: F811
    result = run(config, tmp_path / "cache", sources())
    at = dt.datetime(2026, 11, 7, 1, 0, tzinfo=dt.UTC)
    md, trades_csv = write_walk_forward(
        result, "swing-scanner walk-forward ...", "abc", at, tmp_path
    )
    assert md.name == "20261107-010000_ma_pullback_v1_walkforward.md"
    text = md.read_text(encoding="utf-8")
    assert "| 学習期間 | 評価期間 |" in text and "評価期間をつないだ成績" in text
    assert "2023-01-01〜2024-12-31 | 2025-01-01〜2025-12-31" in text
    lines = trades_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("window,symbol,")
    assert len(lines) - 1 == len(result.test_trades)
    small = [w for w in result.windows if w.test and len(w.test.trades) < result.min_trades]
    assert ("サンプル不足" in text) == bool(small)


def test_cli_walk_forward(tmp_path: Path, capsys) -> None:
    data = tmp_path / "csv"
    data.mkdir()
    for i, code in enumerate(JP + ["1306"]):
        frame(i if code != "1306" else 99).to_csv(data / f"{code}.csv", index=False)
    common = ["--pattern", "ma_pullback", "--market", "JP", "--data-dir", str(data)]
    common += ["--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "out")]
    assert main(["walk-forward", *common, "--start", "2023-01-01", "--end", "2026-10-30"]) == 0
    assert "窓1 2025-01-01〜2025-12-31" in capsys.readouterr().out
    assert main(["walk-forward", *common, "--start", "2026-01-01", "--end", "2026-10-30"]) == 1


@pytest.mark.parametrize("months", [(0, 12, 12), (24, 0, 12), (24, 12, 0)])
def test_invalid_window_lengths_are_rejected(months) -> None:
    with pytest.raises(ValueError):
        make_windows(D(2023, 1, 1), D(2026, 1, 1), *months)
