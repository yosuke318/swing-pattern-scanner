"""日次処理（基本設計書 5.2）。手順3以降（取得・判定・通知）は [1-10] で繋ぐ。"""

import datetime as dt
from pathlib import Path

from swing_scanner.config import Market
from swing_scanner.state import RunRecord, append_run
from swing_scanner.trading_calendar import TargetDay, TradingCalendar, resolve_target


def start_run(
    market: Market,
    calendar: TradingCalendar,
    state_dir: Path,
    now: dt.datetime,
    commit_sha: str,
    override: dt.date | None = None,
) -> TargetDay:
    """手順2: 対象日を決める。休場日なら runs.csv に skipped_holiday を記録する。

    カレンダーを取得できなければ CalendarError をそのまま上げる（呼び出し側で異常終了にする）。
    """
    target = resolve_target(market, calendar, now, override)
    if not target.is_trading_day:
        append_run(
            state_dir,
            RunRecord(
                run_at=now,
                market=market,
                target_date=target.date,
                status="skipped_holiday",
                symbols_ok=0,
                symbols_failed=0,
                warnings=[],
                commit_sha=commit_sha,
            ),
        )
    return target
