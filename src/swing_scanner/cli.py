import argparse
import datetime as dt
import sys
from pathlib import Path

from swing_scanner.config import ConfigError, load_config
from swing_scanner.daily import DailyError, run_daily
from swing_scanner.notify.message import email_body, subject
from swing_scanner.notify.send import NotifyError
from swing_scanner.provenance import commit_sha
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
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
