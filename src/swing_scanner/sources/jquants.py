"""J-Quants API のクライアント（認証とページング）。

認証情報は環境変数から読む。上から順に使う:
  JQUANTS_API_KEY                    v2（API キーをヘッダに付ける）
  JQUANTS_REFRESH_TOKEN              v1（リフレッシュトークンは1週間で切れる）
  JQUANTS_MAIL + JQUANTS_PASSWORD    v1（無人実行で期限切れを起こさない）

v1 は実行のたびに ID トークンを取り直す（有効期限24時間のため、使い回さない）。
v2 のパスと応答のキーは [1-0] の実地調査で確定させる。
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Mapping
from typing import Any, Literal

from swing_scanner.sources.http import ApiError, request_json

ApiVersion = Literal["v1", "v2"]

BASE_URLS: dict[ApiVersion, str] = {
    "v1": "https://api.jquants.com/v1",
    "v2": "https://api.jquants.com/v2",
}


class JQuantsClient:
    def __init__(self, version: ApiVersion, headers: dict[str, str]) -> None:
        self.version = version
        self.base = BASE_URLS[version]
        self.headers = headers

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
