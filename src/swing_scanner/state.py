"""state/ 配下の状態ファイルの読み書き（基本設計書 3.4）。

- results/YYYY-MM.csv  判定ログ（通知しなかった行も含めて全件、判定対象日の月で分ける）。
                       再実行したら (date, market, symbol, pattern_id, version) で置き換える
- notified.json        通知履歴。"<symbol>|<pattern_id>" → 最終通知日
- runs.csv             実行ログ

日本株と米国株のジョブはどちらも同じファイルを書くため、GitHub Actions 側で
concurrency を共通のグループにして直列化する（[1-9]）。
"""

import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

RunStatus = Literal["success", "skipped_holiday", "partial", "failed"]


@dataclass(frozen=True)
class ResultRow:
    date: dt.date
    market: str
    symbol: str
    pattern_id: str
    version: int
    matched: bool
    score: float
    # 条件ごとの合否・実測値・閾値。JSON 文字列にして1列に入れる
    breakdown: list[dict[str, Any]]
    excluded_reason: str
    stop: float | None
    target: float | None
    max_holding_days: int
    notified: bool
    commit_sha: str
    evaluated_at: dt.datetime


@dataclass(frozen=True)
class RunRecord:
    run_at: dt.datetime
    market: str
    target_date: dt.date
    status: RunStatus
    symbols_ok: int
    symbols_failed: int
    warnings: list[str]
    commit_sha: str


RESULT_COLUMNS = [f.name for f in fields(ResultRow)]
RUN_COLUMNS = [f.name for f in fields(RunRecord)]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dt.datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _append_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(v) for k, v in row.items()})


def results_path(state_dir: Path, target_date: dt.date) -> Path:
    return state_dir / "results" / f"{target_date:%Y-%m}.csv"


# 判定ログの1行を特定する列。同じキーの行は再実行で置き換える
RESULT_KEY = ("date", "market", "symbol", "pattern_id", "version")


def upsert_results(state_dir: Path, rows: list[ResultRow]) -> None:
    """判定ログに追記する。同じ RESULT_KEY の行があれば置き換える（同じ対象日の再実行）。"""
    by_month: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        cells = {k: _cell(v) for k, v in asdict(row).items()}
        by_month.setdefault(results_path(state_dir, row.date), []).append(cells)
    for path, new_rows in by_month.items():
        new_keys = {tuple(r[k] for k in RESULT_KEY) for r in new_rows}
        kept: list[dict[str, str]] = []
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                kept = [
                    r for r in csv.DictReader(f) if tuple(r[k] for k in RESULT_KEY) not in new_keys
                ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(kept + new_rows)


def append_run(state_dir: Path, record: RunRecord) -> None:
    _append_csv(state_dir / "runs.csv", RUN_COLUMNS, [asdict(record)])


def notified_key(symbol: str, pattern_id: str) -> str:
    return f"{symbol}|{pattern_id}"


def load_notified(state_dir: Path) -> dict[str, dt.date]:
    path = state_dir / "notified.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: dt.date.fromisoformat(v) for k, v in data.items()}


def save_notified(state_dir: Path, notified: dict[str, dt.date]) -> None:
    path = state_dir / "notified.json"
    data = {k: v.isoformat() for k, v in sorted(notified.items())}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
