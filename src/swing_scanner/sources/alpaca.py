"""Alpaca API のクライアント。

認証情報は環境変数 APCA_API_KEY_ID / APCA_API_SECRET_KEY から読む。
Trading API の接続先は APCA_API_BASE_URL で変えられる（既定はペーパー口座）。

日足（[1-5]）は Market Data API の bars を使う:
- adjustment=split（既定の raw は分割調整なしなので必ず指定する）
- feed は既定で sip（全米の出来高）。無料プランでも15分より古い履歴は SIP を取れる。
  通らなければ環境変数 APCA_DATA_FEED=iex で IEX（全米の出来高の約2.5%）に切り替える
- 売買代金は API が返さないので、VWAP × 出来高で求める（VWAP が無ければ 終値 × 出来高）
"""

from __future__ import annotations

import datetime as dt
import os
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from swing_scanner.sources.base import SourceUnavailable
from swing_scanner.sources.http import ApiError, request_json

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
DEFAULT_FEED = "sip"
NY = ZoneInfo("America/New_York")


class AlpacaClient:
    def __init__(self, key_id: str, secret_key: str, trading_url: str = PAPER_TRADING_URL) -> None:
        self.trading_url = trading_url.rstrip("/")
        self.headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> AlpacaClient:
        key_id, secret = env.get("APCA_API_KEY_ID"), env.get("APCA_API_SECRET_KEY")
        if not (key_id and secret):
            raise ApiError(
                "Alpaca の認証情報が環境変数にありません（APCA_API_KEY_ID / APCA_API_SECRET_KEY）"
            )
        return cls(key_id, secret, env.get("APCA_API_BASE_URL") or PAPER_TRADING_URL)

    def get_trading(self, path: str, params: dict[str, str]) -> Any:
        url = f"{self.trading_url}/v2{path}?{urllib.parse.urlencode(params)}"
        return request_json(url, headers=self.headers)

    def get_data(self, path: str, params: dict[str, str]) -> Any:
        url = f"{DATA_URL}/v2{path}?{urllib.parse.urlencode(params)}"
        return request_json(url, headers=self.headers)


# --- 日足（[1-5]） -----------------------------------------------------------------

DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "trading_value"]


def parse_bars(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """bars を DAILY_COLUMNS の DataFrame にする。日足の t（UTC）は NY 時間の日付にする。"""
    records = []
    for bar in bars:
        try:
            t = dt.datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00"))
            close, volume = float(bar["c"]), float(bar["v"])
            vwap = bar.get("vw")
            records.append(
                {
                    "date": t.astimezone(NY).date(),
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": close,
                    "volume": volume,
                    "trading_value": (float(vwap) if vwap else close) * volume,
                }
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ApiError(f"Alpaca の bars を解釈できません: {bar!r}") from e
    if not records:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = pd.DataFrame(records)
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


class AlpacaDailySource:
    """米国株の日足（OhlcvSource の Alpaca 実装）。認証情報は最初の取得のときに読む。"""

    def __init__(
        self,
        client_factory: Callable[[], AlpacaClient] = AlpacaClient.from_env,
        feed: str | None = None,
    ) -> None:
        self._factory = client_factory
        self._client: AlpacaClient | None = None
        self.feed = feed or os.environ.get("APCA_DATA_FEED") or DEFAULT_FEED

    @property
    def client(self) -> AlpacaClient:
        if self._client is None:
            try:
                self._client = self._factory()
            except ApiError as e:
                raise SourceUnavailable(str(e)) from e
        return self._client

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        client = self.client
        params = {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": start.isoformat(),
            # 日付だけの end がその日の始まりと解釈されても対象日の足が入るよう、1日先まで取って切る
            "end": (end + dt.timedelta(days=1)).isoformat(),
            "adjustment": "split",
            "feed": self.feed,
            "limit": "10000",
        }
        bars: list[dict[str, Any]] = []
        while True:
            data = client.get_data("/stocks/bars", params)
            if not isinstance(data, dict):
                raise ApiError("Alpaca の bars が想定外の応答です")
            bars += (data.get("bars") or {}).get(symbol, [])
            token = data.get("next_page_token")
            if not token:
                break
            params = {**params, "page_token": token}
        df = parse_bars(bars)
        return df[df["date"] <= end].reset_index(drop=True) if not df.empty else df
