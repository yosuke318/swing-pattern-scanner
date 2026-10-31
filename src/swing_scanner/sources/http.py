"""外部APIを叩く最小限の HTTP ヘルパー（標準ライブラリだけで動かす）。"""

import json
import urllib.error
import urllib.request
from typing import Any


class ApiError(Exception):
    """外部APIの呼び出しの失敗（通信エラー、HTTPエラー、JSONでない応答、認証情報の欠落）。"""


def _safe(url: str) -> str:
    # クエリに認証情報（J-Quants v1 の refreshtoken）が入ることがあるので、エラー文には出さない
    return url.split("?", 1)[0]


def request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: Any = None,
    timeout: float = 60.0,
) -> Any:
    hdrs = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode("utf-8", errors="replace")
        raise ApiError(f"{method} {_safe(url)}: HTTP {e.code} {detail}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ApiError(f"{method} {_safe(url)}: {e}") from e
    try:
        return json.loads(raw)
    except ValueError as e:
        raise ApiError(f"{method} {_safe(url)}: JSON ではない応答") from e
