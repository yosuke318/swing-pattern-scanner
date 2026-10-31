"""日次処理（基本設計書 5.2）。手順3以降（取得・判定・通知）は [1-10] で繋ぐ。"""

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

from swing_scanner.config import Config, Market, Pattern
from swing_scanner.filters import FilterOutcome
from swing_scanner.notify.cooldown import prune, record_notified
from swing_scanner.notify.message import Report
from swing_scanner.notify.send import EmailChannel, LineChannel, deliver
from swing_scanner.rules.engine import Evaluation
from swing_scanner.state import ResultRow, RunRecord, append_run, load_notified, save_notified
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


def build_result_row(
    date: dt.date,
    market: Market,
    symbol: str,
    pattern: Pattern,
    evaluation: Evaluation,
    outcome: FilterOutcome,
    notified: bool,
    commit_sha: str,
    evaluated_at: dt.datetime,
) -> ResultRow:
    """判定ログ（state/results/）の1行を作る（手順8）。

    excluded_reason には、通知しなかった理由をまとめて入れる: レジーム対象外と除外フィルタ。
    複数あれば "; " でつなぐ。commit_sha とパターンのバージョンから当時の判定を再現できる。
    """
    reasons = [f"レジーム対象外: {evaluation.skip_reason}"] if evaluation.skip_reason else []
    reasons += outcome.excluded
    plan = evaluation.exit
    return ResultRow(
        date=date,
        market=market,
        symbol=symbol,
        pattern_id=evaluation.pattern_id,
        version=evaluation.version,
        matched=evaluation.matched,
        score=evaluation.score,
        breakdown=evaluation.breakdown(),
        excluded_reason="; ".join(reasons),
        stop=None if plan is None else round(plan.stop, 4),
        target=None if plan is None else round(plan.target, 4),
        max_holding_days=pattern.exit.max_holding_days,
        notified=notified,
        commit_sha=commit_sha,
        evaluated_at=evaluated_at,
    )


def send_report(
    report: Report,
    config: Config,
    state_dir: Path,
    sessions: Sequence[dt.date] | None = None,
    email: EmailChannel | None = None,
    line: LineChannel | None = None,
) -> list[str]:
    """手順10〜11の前半: 通知を送り、送れたら notified.json を更新する。

    送信してから記録する。送信に失敗したら NotifyError がそのまま上がり、notified.json は
    更新しない（再実行で送り直せる。逆の順序だと送れていないのに送信済みになりシグナルが消える）。
    記録のついでに、クールダウンを過ぎたエントリを消す。
    """
    sent = deliver(report, config.settings.notification, email, line)
    days = config.settings.notification.cooldown_business_days
    notified = record_notified(
        load_notified(state_dir),
        [(j.symbol, j.pattern.pattern_id) for j in report.signals],
        report.target_date,
    )
    save_notified(state_dir, prune(notified, report.target_date, days, sessions))
    return sent
