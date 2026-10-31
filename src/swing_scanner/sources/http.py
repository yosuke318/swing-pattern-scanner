"""外部APIを叩く最小限の HTTP ヘルパー（標準ライブラリだけで動かす）。

リトライ方針（[1-6]）: GET だけを、通信エラー・タイムアウト・429・5xx のときに再試行する。
最大 RETRIES 回まで、待ち時間は 2 秒・4 秒（429 は 10 秒・20 秒。J-Quants は超過が続くと
約5分ブロックするので長めに待つ）。認証エラーなどその他の 4xx はすぐ諦める。POST（認証、
LINE の送信）は二重に実行されうるので再試行しない。
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

RETRIES = 2
BACKOFF_SEC = (2.0, 4.0)
RATE_LIMIT_BACKOFF_SEC = (10.0, 20.0)


class ApiError(Exception):
    """外部APIの呼び出しの失敗（通信エラー、HTTPエラー、JSONでない応答、認証情報の欠落）。"""


def _safe(url: str) -> str:
    # クエリに認証情報（J-Quants v1 の refreshtoken）が入ることがあるので、エラー文には出さない
    return url.split("?", 1)[0]


def _retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, urllib.error.URLError | TimeoutError)


def request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: Any = None,
    timeout: float = 60.0,
    retries: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    hdrs = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    attempts = 1 + (retries if retries is not None else (RETRIES if method == "GET" else 0))
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            break
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt + 1 < attempts and _retryable(e):
                is_429 = isinstance(e, urllib.error.HTTPError) and e.code == 429
                waits = RATE_LIMIT_BACKOFF_SEC if is_429 else BACKOFF_SEC
                sleep(waits[min(attempt, len(waits) - 1)])
                continue
            if isinstance(e, urllib.error.HTTPError):
                detail = e.read()[:200].decode("utf-8", errors="replace")
                raise ApiError(f"{method} {_safe(url)}: HTTP {e.code} {detail}") from e
            raise ApiError(f"{method} {_safe(url)}: {e}") from e
    try:
        return json.loads(raw)
    except ValueError as e:
        raise ApiError(f"{method} {_safe(url)}: JSON ではない応答") from e
