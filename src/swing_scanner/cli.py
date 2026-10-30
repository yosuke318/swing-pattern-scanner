import argparse
import datetime as dt
import sys
from pathlib import Path

from swing_scanner.config import ConfigError, load_config
from swing_scanner.daily import start_run
from swing_scanner.provenance import commit_sha
from swing_scanner.trading_calendar import CalendarError, calendar_from_env


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
    try:
        load_config(args.config_dir)
    except ConfigError as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 1
    try:
        target = start_run(
            args.market,
            calendar_from_env(args.market),
            args.state_dir,
            now=dt.datetime.now(dt.UTC),
            commit_sha=commit_sha(),
            override=args.date,
        )
    except CalendarError as e:
        print(f"取引カレンダーで対象日を判定できません: {e}", file=sys.stderr)
        return 1
    if not target.is_trading_day:
        print(f"{target.market} {target.date} は休場日のため終了します")
        return 0
    print(f"{target.market} 対象日 {target.date}")
    # 手順3以降（取得・指標計算・判定・通知）は [1-10] で繋ぐ
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
    run.set_defaults(func=_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
