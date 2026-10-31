"""Alpaca API のクライアント。

認証情報は環境変数 APCA_API_KEY_ID / APCA_API_SECRET_KEY から読む。
Trading API の接続先は APCA_API_BASE_URL で変えられる（既定はペーパー口座）。
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Mapping
from typing import Any

from swing_scanner.sources.http import ApiError, request_json

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"


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
