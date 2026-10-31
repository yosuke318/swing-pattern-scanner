import csv
import datetime as dt
import json
from pathlib import Path

from swing_scanner.provenance import commit_sha
from swing_scanner.state import (
    RESULT_COLUMNS,
    RUN_COLUMNS,
    ResultRow,
    RunRecord,
    append_run,
    load_notified,
    notified_key,
    save_notified,
    upsert_results,
)

REPO_STATE = Path(__file__).resolve().parents[1] / "state"


def _row(day: dt.date, **kw) -> ResultRow:
    base = dict(
        date=day,
        market="JP",
        symbol="7203",
        pattern_id="golden_cross_uptrend",
        version=1,
        matched=True,
        score=75.0,
        breakdown=[{"key": "adx14", "op": ">=", "threshold": 20, "actual": 24.1, "passed": True}],
        excluded_reason="",
        stop=2810.5,
        target=3105.0,
        max_holding_days=10,
        notified=True,
        commit_sha="abc123",
        evaluated_at=dt.datetime(2026, 10, 30, 18, 5, tzinfo=dt.UTC),
    )
    base.update(kw)
    return ResultRow(**base)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_results_are_split_by_month(tmp_path: Path):
    upsert_results(tmp_path, [_row(dt.date(2026, 10, 30)), _row(dt.date(2026, 11, 2))])
    upsert_results(tmp_path, [_row(dt.date(2026, 10, 31), symbol="6758", stop=None)])

    october = _read(tmp_path / "results/2026-10.csv")
    assert [r["symbol"] for r in october] == ["7203", "6758"]
    assert list(october[0]) == RESULT_COLUMNS
    assert october[0]["matched"] == "true"
    assert october[0]["breakdown"].startswith('[{"key":"adx14"')
    assert october[0]["evaluated_at"] == "2026-10-30T18:05:00+00:00"
    assert october[1]["stop"] == ""
    assert len(_read(tmp_path / "results/2026-11.csv")) == 1


def test_rerun_replaces_rows_with_same_key(tmp_path: Path):
    day = dt.date(2026, 10, 30)
    upsert_results(tmp_path, [_row(day), _row(day, symbol="6758"), _row(day, version=2)])
    upsert_results(tmp_path, [_row(day, score=50.0, matched=False)])
    upsert_results(tmp_path, [_row(dt.date(2026, 10, 29))])
    rows = _read(tmp_path / "results/2026-10.csv")
    assert [(r["date"], r["symbol"], r["version"], r["score"]) for r in rows] == [
        ("2026-10-30", "6758", "1", "75.0"),
        ("2026-10-30", "7203", "2", "75.0"),
        ("2026-10-30", "7203", "1", "50.0"),
        ("2026-10-29", "7203", "1", "75.0"),
    ]
    assert rows[2]["matched"] == "false"


def test_results_keep_commas_and_quotes_in_breakdown(tmp_path: Path):
    day = dt.date(2026, 10, 30)
    breakdown = [{"key": "adx14", "note": 'a, "b"'}]
    upsert_results(tmp_path, [_row(day, breakdown=breakdown, excluded_reason="流動性; 決算")])
    upsert_results(tmp_path, [_row(day, symbol="6758")])
    first = _read(tmp_path / "results/2026-10.csv")[0]
    assert json.loads(first["breakdown"]) == breakdown
    assert first["excluded_reason"] == "流動性; 決算"


def test_run_log_matches_committed_header(tmp_path: Path):
    header = (REPO_STATE / "runs.csv").read_text(encoding="utf-8").strip().split(",")
    assert header == RUN_COLUMNS

    record = RunRecord(
        run_at=dt.datetime(2026, 10, 30, 18, 0, tzinfo=dt.UTC),
        market="JP",
        target_date=dt.date(2026, 10, 30),
        status="partial",
        symbols_ok=1,
        symbols_failed=1,
        warnings=["6758: 出来高0"],
        commit_sha="abc123",
    )
    append_run(tmp_path, record)
    append_run(tmp_path, record)
    rows = _read(tmp_path / "runs.csv")
    assert len(rows) == 2
    assert rows[0]["warnings"] == '["6758: 出来高0"]'


def test_notified_round_trip(tmp_path: Path):
    assert load_notified(tmp_path) == {}
    data = {notified_key("NVDA", "golden_cross_uptrend"): dt.date(2026, 10, 29)}
    save_notified(tmp_path, data)
    assert load_notified(tmp_path) == data
    assert load_notified(REPO_STATE) == {}


def test_commit_sha_prefers_github_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    assert commit_sha() == "deadbeef"
