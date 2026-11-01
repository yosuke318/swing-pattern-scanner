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
    """取得元が使えない（認証情報が無い・認証に失敗した）。銘柄単位の失敗ではなく、実行全体を止める。"""


class CsvSource:
    """<data_dir>/<symbol>.csv から読む（列は OhlcvSource と同じ）。動作確認・パターン調整用。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        df = pd.read_csv(self.data_dir / f"{symbol}.csv", dtype={"date": str})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def source_for(market: Market, data_dir: Path | None = None) -> OhlcvSource:
    """日本株は J-Quants、米国株は Alpaca。data_dir を渡すと CSV から読む。"""
    if data_dir is not None:
        return CsvSource(data_dir)
    # 取得元のモジュールがこのモジュールを読むので、循環しないようここで読み込む
    if market == "JP":
        from swing_scanner.sources.jquants import JQuantsDailySource

        return JQuantsDailySource()
    from swing_scanner.sources.alpaca import AlpacaDailySource

    return AlpacaDailySource()
