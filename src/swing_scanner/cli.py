import argparse
import datetime as dt
import sys
from pathlib import Path

from swing_scanner.backtest.cache import DEFAULT_CACHE_DIR, OhlcvCache
from swing_scanner.backtest.report import DEFAULT_REPORT_DIR, summarize, write_report
from swing_scanner.backtest.runner import run_backtest
from swing_scanner.backtest.walk_forward import run_walk_forward, write_walk_forward
from swing_scanner.config import ConfigError, Symbol, load_config
from swing_scanner.daily import DailyError, run_daily
from swing_scanner.notify.message import email_body, subject
from swing_scanner.notify.send import NotifyError
from swing_scanner.provenance import commit_sha, has_uncommitted_changes
from swing_scanner.sources.base import SourceUnavailable, source_for
from swing_scanner.trading_calendar import CalendarError, WeekdayCalendar, calendar_from_env


def _validate_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config_dir)
    except ConfigError as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 1
    watched = sum(1 for s in config.symbols if s.watched)
    print(
        f"OK: 監視銘柄 {watched}/{len(config.symbols)}、イベント {len(config.events)}、"
        f"パターン {len(config.patterns)}（有効 {len(config.active_patterns())}）"
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    """日次処理。正常（休場日を含む）は 0、異常は 1（GitHub Actions のジョブ失敗通知の起点）。"""
    try:
        config = load_config(args.config_dir)
        calendar = (
            WeekdayCalendar(args.market)
            if args.calendar == "weekdays"
            else calendar_from_env(args.market)
        )
        result = run_daily(
            config,
            args.market,
            calendar,
            source_for(args.market, args.data_dir),
            args.state_dir,
            now=dt.datetime.now(dt.UTC),
            commit_sha=commit_sha(),
            override=args.date,
            dry_run=args.dry_run,
            force=args.force,
        )
    except ConfigError as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 1
    except CalendarError as e:
        print(f"取引カレンダーで対象日を判定できません: {e}", file=sys.stderr)
        return 1
    except (DailyError, SourceUnavailable, NotifyError) as e:
        print(f"日次処理を中断しました: {e}", file=sys.stderr)
        return 1
    if args.dry_run and result.report is not None:
        print(f"件名: {subject(result.report)}\n")
        print(email_body(result.report))
    return 0


def _backtest(args: argparse.Namespace) -> int:
    """バックテスト。過去データを .cache/ に保存し、レポートを reports/backtest/ に出す。"""
    try:
        config = load_config(args.config_dir)
        pattern = config.pattern(args.pattern, args.version)
        symbols = _symbols_for(config, args)
    except (ConfigError, ValueError) as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 1
    if args.start > args.end:
        print("--start は --end 以前の日付にしてください", file=sys.stderr)
        return 1
    cache = OhlcvCache(source_for(args.market, args.data_dir), args.cache_dir)
    try:
        result = run_backtest(
            config, pattern, args.market, symbols, args.start, args.end, cache, args.refresh_cache
        )
    except SourceUnavailable as e:
        print(f"バックテストを中断しました: {e}", file=sys.stderr)
        return 1
    now = dt.datetime.now(dt.UTC)
    command = "swing-scanner " + " ".join(args.argv)
    commit = commit_sha() + ("（未コミットの変更あり）" if has_uncommitted_changes() else "")
    md, trades_csv = write_report(result, config, command, commit, now, args.out_dir)
    s = summarize(result)
    win_rate = "—" if s.win_rate is None else f"{s.win_rate * 100:.1f}%"
    print(f"シグナル {s.signals}件、トレード {s.trades}件、勝率 {win_rate}")
    print(f"レポート: {md}")
    print(f"トレード一覧: {trades_csv}")
    failed = [d.symbol for d in result.data if d.error]
    if failed:
        print(f"データを使えなかった銘柄: {', '.join(failed)}（レポートの「データ」を参照）")
    return 0


def _symbols_for(config, args: argparse.Namespace) -> list[Symbol]:
    if not args.symbols:
        return config.watched_symbols(args.market)
    known = {s.symbol: s for s in config.symbols if s.market == args.market}
    return [
        known.get(code) or Symbol(symbol=code, name=code, market=args.market, watched=False)
        for code in args.symbols.split(",")
    ]


def _walk_forward(args: argparse.Namespace) -> int:
    """ウォークフォワード検証。学習期間で閾値と出口を選び、その後ろの評価期間で成績を測る。"""
    try:
        config = load_config(args.config_dir)
        pattern = config.pattern(args.pattern, args.version)
        symbols = _symbols_for(config, args)
    except (ConfigError, ValueError) as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 1
    cache = OhlcvCache(source_for(args.market, args.data_dir), args.cache_dir)
    try:
        result = run_walk_forward(
            config,
            pattern,
            args.market,
            symbols,
            args.start,
            args.end,
            cache,
            args.train_months,
            args.test_months,
            args.step_months,
            args.refresh_cache,
        )
    except ValueError as e:
        print(f"期間の指定が不正です: {e}", file=sys.stderr)
        return 1
    except SourceUnavailable as e:
        print(f"検証を中断しました: {e}", file=sys.stderr)
        return 1
    now = dt.datetime.now(dt.UTC)
    command = "swing-scanner " + " ".join(args.argv)
    commit = commit_sha() + ("（未コミットの変更あり）" if has_uncommitted_changes() else "")
    md, trades_csv = write_walk_forward(result, command, commit, now, args.out_dir)
    for i, w in enumerate(result.windows, 1):
        train, test = w.train_summary, w.test_summary
        picked = w.params.label() if w.params else "選べない（件数不足）"
        counts = f"学習 {train.trades if train else 0}件 / 評価 {test.trades if test else 0}件"
        print(f"窓{i} {w.window.test_start}〜{w.window.test_end}: {picked}（{counts}）")
    print(f"レポート: {md}")
    print(f"評価期間のトレード一覧: {trades_csv}")
    return 0


def _add_backtest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pattern", required=True, help="pattern_id")
    parser.add_argument("--version", type=int, help="省略すると active なバージョン")
    parser.add_argument("--market", choices=["JP", "US"], required=True)
    parser.add_argument("--symbols", help="カンマ区切りの銘柄コード。省略すると監視中の銘柄")
    parser.add_argument(
        "--refresh-cache", action="store_true", help="キャッシュを使わず API から取り直す"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="日足を API ではなく <DIR>/<銘柄コード>.csv から読む"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swing-scanner")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config", help="config/ を読み込んでスキーマ検証する").set_defaults(
        func=_validate_config
    )
    run = sub.add_parser("run", help="日次処理を実行する")
    run.add_argument("--market", choices=["JP", "US"], required=True)
    run.add_argument(
        "--date",
        type=dt.date.fromisoformat,
        help="対象日（YYYY-MM-DD）。省略時は現地時刻で直近に引けた日",
    )
    run.add_argument("--state-dir", type=Path, default=Path("state"))
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="通知を送らず、state も更新しない（通知文を表示する）",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="同じ対象日の通知を送り直す（送信済みのシグナルも同じ日の分は再送する）",
    )
    run.add_argument(
        "--data-dir", type=Path, help="日足を API ではなく <DIR>/<銘柄コード>.csv から読む"
    )
    run.add_argument(
        "--calendar",
        choices=["api", "weekdays"],
        default="api",
        help="weekdays: API を使わず土日だけを休場とする（動作確認用）",
    )
    run.set_defaults(func=_run)

    bt = sub.add_parser("backtest", help="過去データでパターンを検証する")
    _add_backtest_args(bt)
    bt.add_argument("--start", type=dt.date.fromisoformat, required=True, help="評価期間の開始日")
    bt.add_argument("--end", type=dt.date.fromisoformat, required=True, help="評価期間の終了日")
    bt.set_defaults(func=_backtest)

    wf = sub.add_parser("walk-forward", help="学習期間と評価期間を分けて検証する")
    _add_backtest_args(wf)
    wf.add_argument(
        "--start", type=dt.date.fromisoformat, required=True, help="最初の学習期間の開始日"
    )
    wf.add_argument(
        "--end", type=dt.date.fromisoformat, required=True, help="最後の評価期間の終了日"
    )
    wf.add_argument("--train-months", type=int, default=24, help="学習期間（か月、既定 24）")
    wf.add_argument("--test-months", type=int, default=12, help="評価期間（か月、既定 12）")
    wf.add_argument("--step-months", type=int, default=12, help="窓をずらす幅（か月、既定 12）")
    wf.set_defaults(func=_walk_forward)

    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
