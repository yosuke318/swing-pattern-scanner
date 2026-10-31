"""日足 OHLCV の取得元の共通インターフェース（[1-4] J-Quants、[1-5] Alpaca が実装する）。"""

import datetime as dt
from pathlib import Path
from typing import Protocol

import pandas as pd

from swing_scanner.config import Market

OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class OhlcvSource(Protocol):
    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """start〜end（両端を含む）の日足。列は date / open / high / low / close / volume
        （任意で trading_value）、日付の昇順。取得に失敗したら ApiError か OSError を上げる。"""
        ...


class SourceUnavailable(Exception):
    """取得元がまだ使えない（未実装・設定不足）。銘柄単位の失敗ではなく、実行全体を止める。"""


class UnavailableSource:
    def __init__(self, market: Market) -> None:
        self.market = market

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        name = {"JP": "日本株（J-Quants、[1-4]）", "US": "米国株（Alpaca、[1-5]）"}[self.market]
        raise SourceUnavailable(
            f"{name}の取得はまだ実装されていません。--data-dir で CSV を渡してください"
        )


class CsvSource:
    """<data_dir>/<symbol>.csv から読む（列は OhlcvSource と同じ）。動作確認・パターン調整用。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / f"{symbol}.csv", dtype={"date": str})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def source_for(market: Market, data_dir: Path | None = None) -> OhlcvSource:
    if data_dir is not None:
        return CsvSource(data_dir)
    return UnavailableSource(market)
