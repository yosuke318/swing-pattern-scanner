import argparse
import sys
from pathlib import Path

from swing_scanner.config import ConfigError, load_config


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swing-scanner")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config", help="config/ を読み込んでスキーマ検証する").set_defaults(
        func=_validate_config
    )
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
