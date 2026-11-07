"""バックテスト用の過去データのキャッシュ（.cache/ohlcv/、コミットしない）。

銘柄ごとに Parquet と、取得した範囲・日時のメタデータ（JSON）を置く。要求した範囲を覆っていれば
キャッシュから読み、API を叩かない。覆っていなければ（または refresh なら）取り直す。

キャッシュは調整済み価格なので、取得後に株式分割があると古いキャッシュと食い違う。fetched_at を
レポートに残し、ときどき refresh で取り直す。
"""

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from swing_scanner.config import Market
from swing_scanner.sources.base import OhlcvSource

DEFAULT_CACHE_DIR = Path(".cache/ohlcv")


@dataclass(frozen=True)
class CacheEntry:
    data: pd.DataFrame
    # 取得したときに要求した範囲と、取得日時
    start: dt.date
    end: dt.date
    fetched_at: dt.datetime
    from_cache: bool


class OhlcvCache:
    def __init__(
        self,
        source: OhlcvSource,
        root: Path = DEFAULT_CACHE_DIR,
        now: dt.datetime | None = None,
    ) -> None:
        self.source = source
        self.root = root
        self._now = now

    def _paths(self, market: Market, symbol: str) -> tuple[Path, Path]:
        base = self.root / market / symbol.replace("/", "_")
        return base.with_suffix(".parquet"), base.with_suffix(".meta.json")

    def load(
        self,
        market: Market,
        symbol: str,
        start: dt.date,
        end: dt.date,
        refresh: bool = False,
    ) -> CacheEntry:
        data_path, meta_path = self._paths(market, symbol)
        if not refresh and data_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_start = dt.date.fromisoformat(meta["start"])
            cached_end = dt.date.fromisoformat(meta["end"])
            if cached_start <= start and end <= cached_end:
                df = pd.read_parquet(data_path)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
                fetched_at = dt.datetime.fromisoformat(meta["fetched_at"])
                return CacheEntry(df, cached_start, cached_end, fetched_at, from_cache=True)

        df = self.source.fetch(symbol, start, end)
        fetched_at = self._now or dt.datetime.now(dt.UTC)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        meta = {
            "market": market,
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "rows": len(df),
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return CacheEntry(df.reset_index(drop=True), start, end, fetched_at, from_cache=False)
