"""J-Quants API のクライアント（認証とページング）。

認証情報は環境変数から読む。上から順に使う:
  JQUANTS_API_KEY                    v2（API キーをヘッダに付ける）
  JQUANTS_REFRESH_TOKEN              v1（リフレッシュトークンは1週間で切れる）
  JQUANTS_MAIL + JQUANTS_PASSWORD    v1（無人実行で期限切れを起こさない）

v1 は実行のたびに ID トークンを取り直す（有効期限24時間のため、使い回さない）。
v2 の API キーには有効期限が無い。

レート制限はライトプランで 60 リクエスト/分（超えると 429、超過が続くと約5分ブロック）なので、
リクエストの間隔を MIN_INTERVAL_SEC 以上あける。日足は銘柄ごとに from / to で取る（50銘柄で
約50リクエスト。日付ごとに全銘柄を取ると300リクエストかかる）。

v2 の応答のキー（data）とページングは公式ドキュメントからの推測で、[1-0] の実地調査で確かめる。
"""

from __future__ import annotations

import datetime as dt
import math
import os
import time
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any, Literal

import pandas as pd

from swing_scanner.sources.base import SourceUnavailable
from swing_scanner.sources.http import ApiError, request_json

ApiVersion = Literal["v1", "v2"]

BASE_URLS: dict[ApiVersion, str] = {
    "v1": "https://api.jquants.com/v1",
    "v2": "https://api.jquants.com/v2",
}
# 60 リクエスト/分に収める間隔
MIN_INTERVAL_SEC = 1.1


class JQuantsClient:
    def __init__(
        self,
        version: ApiVersion,
        headers: dict[str, str],
        min_interval: float = MIN_INTERVAL_SEC,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.version = version
        self.base = BASE_URLS[version]
        self.headers = headers
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._last_request: float | None = None

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> JQuantsClient:
        api_key = env.get("JQUANTS_API_KEY")
        if api_key:
            return cls("v2", {"x-api-key": api_key})
        return cls("v1", {"Authorization": f"Bearer {_v1_id_token(env)}"})

    def get_all(
        self, path: str, params: dict[str, str], rows_keys: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        """pagination_key をたどって全ページのレコードを返す。

        rows_keys は応答の中でレコードのリストが入るキーの候補（v1 と v2 で名前が違う）。
        """
        rows: list[dict[str, Any]] = []
        while True:
            url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
            self._throttle()
            data = request_json(url, headers=self.headers)
            if not isinstance(data, dict):
                raise ApiError(f"J-Quants {path}: 想定外の応答です")
            key = next((k for k in rows_keys if k in data), None)
            if key is None:
                raise ApiError(f"J-Quants {path}: 応答に {'/'.join(rows_keys)} がありません")
            rows += data[key]
            pagination_key = data.get("pagination_key")
            if not pagination_key:
                return rows
            params = {**params, "pagination_key": pagination_key}


def _v1_id_token(env: Mapping[str, str]) -> str:
    base = BASE_URLS["v1"]
    refresh = env.get("JQUANTS_REFRESH_TOKEN")
    if not refresh:
        mail, password = env.get("JQUANTS_MAIL"), env.get("JQUANTS_PASSWORD")
        if not (mail and password):
            raise ApiError(
                "J-Quants の認証情報が環境変数にありません"
                "（JQUANTS_API_KEY / JQUANTS_REFRESH_TOKEN / JQUANTS_MAIL + JQUANTS_PASSWORD）"
            )
        data = request_json(
            f"{base}/token/auth_user",
            method="POST",
            body={"mailaddress": mail, "password": password},
        )
        refresh = data["refreshToken"]
    q = urllib.parse.urlencode({"refreshtoken": refresh})
    return request_json(f"{base}/token/auth_refresh?{q}", method="POST")["idToken"]


# --- 株価四本値（[1-4]） -------------------------------------------------------------

DAILY_PATHS: dict[ApiVersion, str] = {"v1": "/prices/daily_quotes", "v2": "/equities/bars/daily"}
DAILY_ROWS_KEYS = ("daily_quotes", "data")
# 分割・併合を調整した列。売買代金は調整の影響を受けないので調整前の列をそのまま使う
ADJUSTED_COLUMNS: dict[ApiVersion, dict[str, str]] = {
    "v1": {
        "open": "AdjustmentOpen",
        "high": "AdjustmentHigh",
        "low": "AdjustmentLow",
        "close": "AdjustmentClose",
        "volume": "AdjustmentVolume",
        "trading_value": "TurnoverValue",
    },
    "v2": {
        "open": "AdjO",
        "high": "AdjH",
        "low": "AdjL",
        "close": "AdjC",
        "volume": "AdjVo",
        "trading_value": "Va",
    },
}
DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "trading_value"]


def to_jquants_code(symbol: str) -> str:
    """4桁の銘柄コードを J-Quants の5桁にする（末尾に 0 を付ける。"7203" → "72030"）。"""
    return f"{symbol}0" if len(symbol) == 4 else symbol


def _number(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def parse_daily_quotes(rows: list[dict[str, Any]], version: ApiVersion) -> pd.DataFrame:
    """株価四本値を、調整後の価格の DataFrame（DAILY_COLUMNS、日付の昇順）にする。

    売買の無い日は四本値・出来高・売買代金が null で返るので、NaN のまま残す（扱いは [1-6]）。
    v2 のストップ高・安のフラグ（UL / LL）は upper_limit / lower_limit 列にする。
    """
    cols = ADJUSTED_COLUMNS[version]
    records = []
    for row in rows:
        try:
            record: dict[str, Any] = {"date": dt.date.fromisoformat(str(row["Date"]))}
            record |= {name: _number(row.get(src)) for name, src in cols.items()}
        except (KeyError, TypeError, ValueError) as e:
            raise ApiError(f"J-Quants の株価四本値を解釈できません: {row!r}") from e
        if "UL" in row or "LL" in row:
            record["upper_limit"] = str(row.get("UL")) == "1"
            record["lower_limit"] = str(row.get("LL")) == "1"
        records.append(record)
    if not records:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    df = pd.DataFrame(records)
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


class JQuantsDailySource:
    """日本株の日足（OhlcvSource の J-Quants 実装）。

    認証は最初の取得のときに行う（休場日には API を叩かない）。認証情報が無い・認証に失敗した
    ときは SourceUnavailable で実行全体を止める（銘柄単位の失敗にしない）。
    """

    def __init__(
        self, client_factory: Callable[[], JQuantsClient] = JQuantsClient.from_env
    ) -> None:
        self._factory = client_factory
        self._client: JQuantsClient | None = None

    @property
    def client(self) -> JQuantsClient:
        if self._client is None:
            try:
                self._client = self._factory()
            except ApiError as e:
                raise SourceUnavailable(f"J-Quants の認証ができません: {e}") from e
        return self._client

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        client = self.client
        params = {"code": to_jquants_code(symbol), "from": start.isoformat(), "to": end.isoformat()}
        rows = client.get_all(DAILY_PATHS[client.version], params, DAILY_ROWS_KEYS)
        return parse_daily_quotes(rows, client.version)
