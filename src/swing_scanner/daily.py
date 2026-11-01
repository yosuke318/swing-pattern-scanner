"""日次処理（基本設計書 5.2 の手順1〜11）。各手順の中身は他のモジュールが持ち、ここはつなぎ方と
失敗時の振る舞いだけを書く。

  1 起動（cli.py）            2 休場日判定        3 監視銘柄の読み込み   4 OHLCV 取得・検証
  5 指標計算                  6 除外フィルタ      7 パターン照合         8 判定ログ
  9 通知対象の抽出            10 通知文・送信     11 state の更新（コミットはワークフロー）

失敗時の振る舞い:
- 銘柄単位の失敗（取得失敗・データの欠損や異常値）はその銘柄だけ外して続け、警告に積む（partial）
- 全銘柄の取得失敗・カレンダーの取得失敗・送信失敗は異常終了する。手順単位の例外は握りつぶさない
  （指標計算や照合が壊れたまま通知するのが最悪の壊れ方なので）
- 送ってから記録する。送信に失敗したら判定ログ・notified.json・runs.csv のどれも更新しない
- 同じ対象日の再実行: 判定ログは置き換え、送信済みのシグナルはクールダウンで弾く。その日の通知を
  すでに送っていて新しいシグナルも無ければ、通知は送らない（force なら送り直す）
- dry_run: 通知を送らず、state も書かない（判定結果と通知文を返すだけ）
"""

import datetime as dt
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from swing_scanner.config import Config, Market, Pattern, Symbol
from swing_scanner.features import compute_features
from swing_scanner.filters import FilterOutcome, apply_filters, missing_earnings_warnings
from swing_scanner.notify.cooldown import prune, record_notified
from swing_scanner.notify.message import Report
from swing_scanner.notify.select import Judgement, select
from swing_scanner.notify.send import EmailChannel, LineChannel, deliver
from swing_scanner.rules.engine import Evaluation, evaluate
from swing_scanner.sources.base import OhlcvSource
from swing_scanner.sources.http import ApiError
from swing_scanner.state import (
    ResultRow,
    RunRecord,
    RunStatus,
    already_reported,
    append_run,
    load_notified,
    notified_key,
    save_notified,
    upsert_results,
)
from swing_scanner.trading_calendar import TargetDay, TradingCalendar, resolve_target
from swing_scanner.validation import ValidationResult, failed, validate_ohlcv

# 取得開始日の目安。lookback_days 営業日を祝日込みで確実に含むよう、暦日に換算して余裕を持たせる
CALENDAR_DAYS_PER_BUSINESS_DAY = 1.6
# 取引カレンダーは取得期間（歯抜けの検証・クールダウン）から対象日の先（決算またぎ）まで取る
SESSIONS_AFTER_DAYS = 45


class DailyError(Exception):
    """手順単位の失敗（全銘柄の取得失敗など）。異常終了にする。"""


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


@contextmanager
def _step(log: Callable[[str], None], label: str) -> Iterator[dict[str, str]]:
    """手順の所要時間と結果をログに出す。with の中で info["result"] に結果を書く。"""
    info: dict[str, str] = {"result": ""}
    started = time.monotonic()
    yield info
    log(f"[{label}] {info['result']}（{time.monotonic() - started:.1f}秒）")


def start_run(
    market: Market,
    calendar: TradingCalendar,
    state_dir: Path,
    now: dt.datetime,
    commit_sha: str,
    override: dt.date | None = None,
    record: bool = True,
) -> TargetDay:
    """手順2: 対象日を決める。休場日なら runs.csv に skipped_holiday を記録する（record のとき）。

    カレンダーを取得できなければ CalendarError をそのまま上げる（呼び出し側で異常終了にする）。
    """
    target = resolve_target(market, calendar, now, override)
    if not target.is_trading_day and record:
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

    excluded_reason には、通知しなかった理由をまとめて入れる: レジーム対象外・必須条件未達と
    除外フィルタ。
    複数あれば "; " でつなぐ。commit_sha とパターンのバージョンから当時の判定を再現できる。
    """
    skip = evaluation.skip_reason
    reasons = []
    if skip:
        reasons.append(skip if skip.startswith("必須条件未達") else f"レジーム対象外: {skip}")
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


# --- 手順1〜11 ---------------------------------------------------------------------


@dataclass(frozen=True)
class Fetched:
    frames: dict[str, pd.DataFrame]
    benchmark: pd.DataFrame | None
    failed: list[str]
    warnings: list[str]
    benchmark_failed: bool = False


@dataclass(frozen=True)
class DailyResult:
    target: TargetDay
    status: RunStatus
    report: Report | None = None
    sent: list[str] = field(default_factory=list)
    result_rows: list[ResultRow] = field(default_factory=list)


def _fetch_one(
    source: OhlcvSource,
    symbol: str,
    day: dt.date,
    lookback: int,
    sessions: Sequence[dt.date],
) -> ValidationResult:
    start = day - dt.timedelta(days=math.ceil(lookback * CALENDAR_DAYS_PER_BUSINESS_DAY))
    try:
        df = source.fetch(symbol, start, day)
    except (ApiError, OSError, ValueError) as e:
        return failed("error", f"取得に失敗しました: {e}")
    if "date" in df:
        df = df[df["date"] <= day].tail(lookback).reset_index(drop=True)
    return validate_ohlcv(df, day, sessions)


def fetch_all(
    source: OhlcvSource,
    symbols: Sequence[Symbol],
    benchmark: str,
    day: dt.date,
    lookback: int,
    sessions: Sequence[dt.date] = (),
) -> Fetched:
    """手順4: 監視銘柄とベンチマークの日足を取り、検証する（[1-6]）。

    検証に通らない銘柄は外して警告に積む。通った銘柄の注意事項（歯抜け・値動きのジャンプなど）も
    警告に積むが、判定には使う。
    """
    frames: dict[str, pd.DataFrame] = {}
    failed_symbols: list[str] = []
    warnings: list[str] = []
    for s in symbols:
        result = _fetch_one(source, s.symbol, day, lookback, sessions)
        label = f"{s.symbol} {s.name}"
        if result.ok:
            frames[s.symbol] = result.data
        else:
            failed_symbols.append(s.symbol)
            warnings.append(f"{label}: {result.reason}（判定から外しました）")
        warnings += [f"{label}: {w}" for w in result.warnings]
    bench = _fetch_one(source, benchmark, day, lookback, sessions)
    if not bench.ok:
        warnings.append(
            f"ベンチマーク {benchmark}: {bench.reason}（相対強度と市場環境を判定できません）"
        )
    warnings += [f"ベンチマーク {benchmark}: {w}" for w in bench.warnings]
    return Fetched(frames, bench.data, failed_symbols, warnings, benchmark_failed=not bench.ok)


def run_daily(
    config: Config,
    market: Market,
    calendar: TradingCalendar,
    source: OhlcvSource,
    state_dir: Path,
    *,
    now: dt.datetime,
    commit_sha: str,
    override: dt.date | None = None,
    dry_run: bool = False,
    force: bool = False,
    email: EmailChannel | None = None,
    line: LineChannel | None = None,
    log: Callable[[str], None] = _stderr,
) -> DailyResult:
    """日次処理を1回分実行する。

    異常時は例外（DailyError / SourceUnavailable / CalendarError / NotifyError）を上げる。
    """
    settings = config.settings

    # 2. 休場日判定
    target = start_run(market, calendar, state_dir, now, commit_sha, override, record=not dry_run)
    if not target.is_trading_day:
        log(f"[手順2] {market} {target.date} は休場日のため終了します")
        return DailyResult(target, "skipped_holiday")
    day = target.date
    log(f"[手順2] {market} 対象日 {day}")
    lookback = settings.fetch.lookback_days
    sessions = sorted(
        calendar.sessions(
            day - dt.timedelta(days=math.ceil(lookback * CALENDAR_DAYS_PER_BUSINESS_DAY)),
            day + dt.timedelta(days=SESSIONS_AFTER_DAYS),
        )
    )

    # 3. 監視銘柄
    symbols = config.watched_symbols(market)
    log(f"[手順3] 監視銘柄 {len(symbols)}件")
    if not symbols:
        raise DailyError(f"{market} の監視銘柄がありません")

    # 4. 取得・検証
    with _step(log, "手順4 取得・検証") as info:
        benchmark_symbol = settings.markets[market].benchmark_symbol
        past_sessions = [d for d in sessions if d <= day]
        fetched = fetch_all(source, symbols, benchmark_symbol, day, lookback, past_sessions)
        info["result"] = f"{len(fetched.frames)}/{len(symbols)}銘柄"
    if not fetched.frames:
        raise DailyError("全銘柄の取得に失敗しました: " + " / ".join(fetched.warnings))

    # 5. 指標計算（例外は握りつぶさない）
    with _step(log, "手順5 指標計算") as info:
        features = {
            sym: compute_features(df, fetched.benchmark, settings.features).iloc[-1]
            for sym, df in fetched.frames.items()
        }
        benchmark_trend = None
        if fetched.benchmark is not None:
            bench = compute_features(fetched.benchmark, None, settings.features)
            benchmark_trend = bench["trend_regime"].iloc[-1]
        info["result"] = f"{len(features)}銘柄、指数の trend_regime={benchmark_trend}"

    # 6〜7. 除外フィルタとパターン照合
    with _step(log, "手順6〜7 除外フィルタ・照合") as info:
        patterns = config.active_patterns()
        judgements = [
            Judgement(
                market,
                s.symbol,
                s.name,
                p,
                evaluate(p, features[s.symbol]),
                apply_filters(
                    config, market, s.symbol, day, p, features[s.symbol], benchmark_trend, sessions
                ),
            )
            for s in symbols
            if s.symbol in features
            for p in patterns
        ]
        info["result"] = f"{len(features)}銘柄 × {len(patterns)}パターン"

    # 9. 通知対象の抽出
    notified = load_notified(state_dir)
    selection = select(judgements, config, notified, day, sessions, resend_same_day=force)
    warnings = fetched.warnings + missing_earnings_warnings(config, market, day)
    report = Report(
        market,
        day,
        selection.signals,
        selection.near_misses,
        warnings,
        {p.pattern_id: config.score_threshold(p) for p in patterns},
    )
    log(
        f"[手順9] シグナル {len(selection.signals)}件、惜しかった銘柄 "
        f"{len(selection.near_misses)}件、警告 {len(warnings)}件"
    )

    # 10〜11. 送信してから記録する
    sent: list[str] = []
    if dry_run:
        log("[手順10] dry-run のため送信しません")
    elif selection.signals or force or not already_reported(state_dir, market, day):
        sent = send_report(report, config, state_dir, sessions, email, line)
        log(f"[手順10] 送信しました: {', '.join(sent) or '（有効なチャネルなし）'}")
    else:
        log("[手順10] この対象日の通知は送信済みで、新しいシグナルも無いため送りません")

    # 8. 判定ログ（通知したかどうかが決まってから書く）
    notified_now = notified if dry_run else load_notified(state_dir)
    evaluated_at = now
    rows = [
        build_result_row(
            day,
            market,
            j.symbol,
            j.pattern,
            j.evaluation,
            j.outcome,
            notified_now.get(notified_key(j.symbol, j.pattern.pattern_id)) == day,
            commit_sha,
            evaluated_at,
        )
        for j in judgements
    ]
    # 判定から外した銘柄があれば partial（注意事項だけなら success）
    status: RunStatus = "partial" if fetched.failed or fetched.benchmark_failed else "success"
    if not dry_run:
        upsert_results(state_dir, rows)
        append_run(
            state_dir,
            RunRecord(
                run_at=now,
                market=market,
                target_date=day,
                status=status,
                symbols_ok=len(fetched.frames),
                symbols_failed=len(fetched.failed),
                warnings=warnings,
                commit_sha=commit_sha,
            ),
        )
        log(f"[手順8・11] 判定ログ {len(rows)}行、runs.csv に {status} を記録しました")
    return DailyResult(target, status, report, sent, rows)
