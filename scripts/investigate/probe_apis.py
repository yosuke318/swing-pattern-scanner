"""[1-0] J-Quants / Alpaca のレスポンス実地調査用スクリプト（使い捨て。本実装は [1-4] [1-5]）。

レスポンスをそのまま .cache/investigate/<実行日時>/ に保存し、観察用の summary.json を作る。
標準ライブラリだけで動く。

認証情報は環境変数から読む:
  J-Quants  JQUANTS_API_KEY（v2）または JQUANTS_REFRESH_TOKEN（v1）
            または JQUANTS_MAIL + JQUANTS_PASSWORD（v1）
  Alpaca    APCA_API_KEY_ID + APCA_API_SECRET_KEY

使い方:
  python scripts/investigate/probe_apis.py all       # 全項目
  python scripts/investigate/probe_apis.py jp-today  # 当日分が取れるか（引け後、時刻を変えて数回）
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

OUT_ROOT = Path(".cache/investigate")
JST = dt.timezone(dt.timedelta(hours=9))

# 値がさ株・低位株・分割実績のある銘柄・指数ETFを混ぜる
JP_SYMBOLS = ["7203", "6758", "8035", "9983", "7201", "4689", "1306"]
US_SYMBOLS = ["AAPL", "NVDA", "GOOGL", "AMZN", "SPY"]

# 分割日をまたぐ期間（調整済み価格の返り方の確認用）
SPLIT_WINDOWS = {
    "JP": [("6758", "2024-09-01", "2024-10-31")],  # ソニーグループ 1:5（2024-10-01）
    "US": [("NVDA", "2024-05-20", "2024-06-28")],  # NVIDIA 1:10（2024-06-10）
}

LOOKBACK_CALENDAR_DAYS = 450  # 約300営業日
FAR_PAST = "2015-01-01"  # 取得可能期間の確認用


class Recorder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.requests: list[dict[str, Any]] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, url: str, headers: dict[str, str], method: str = "GET") -> Any:
        req = urllib.request.Request(url, headers=headers, method=method)
        started = time.monotonic()
        status: int | None = None
        resp_headers: dict[str, str] = {}
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                status = r.status
                resp_headers = dict(r.headers)
                body = r.read()
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = dict(e.headers)
            body = e.read()
        elapsed = time.monotonic() - started

        try:
            data: Any = json.loads(body)
        except ValueError:
            data = body.decode("utf-8", errors="replace")
        idx = len(self.requests)
        path = self.out_dir / f"{idx:03d}_{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        # 認証情報を含むクエリ（refreshtoken）は記録しない
        safe_url = url.split("refreshtoken=")[0]
        self.requests.append(
            {
                "name": name,
                "url": safe_url,
                "status": status,
                "elapsed_sec": round(elapsed, 3),
                "bytes": len(body),
                "rate_limit_headers": {
                    k: v for k, v in resp_headers.items() if "ratelimit" in k.lower()
                },
                "file": path.name,
            }
        )
        print(f"{status} {elapsed:6.2f}s {name}", file=sys.stderr)
        return data


def _rows(data: Any) -> list[dict[str, Any]]:
    """レスポンスからレコードのリストを取り出す（形が分からないので、最初に見つかったリスト）。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                return v
            if isinstance(v, dict):
                inner = _rows(v)
                if inner:
                    return inner
    return []


def _profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    keys = sorted({k for r in rows for k in r})
    nulls = {k: sum(1 for r in rows if r.get(k) in (None, "")) for k in keys}
    dates = [str(r.get("Date") or r.get("t") or r.get("date") or "") for r in rows]
    return {
        "rows": len(rows),
        "keys": keys,
        "null_counts": {k: v for k, v in nulls.items() if v},
        "first_date": min(dates),
        "last_date": max(dates),
        "sample_first": rows[0],
        "sample_last": rows[-1],
    }


# --- J-Quants ---------------------------------------------------------------


class JQuants:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec
        api_key = os.environ.get("JQUANTS_API_KEY")
        if api_key:
            # v2: API キーをヘッダに付ける方式。パスは公式ドキュメントで要確認
            self.version = "v2"
            self.base = "https://api.jquants.com/v2"
            self.headers = {"x-api-key": api_key}
            self.daily_path = "/equities/bars/daily"
            self.calendar_path = "/markets/calendar"
        else:
            self.version = "v1"
            self.base = "https://api.jquants.com/v1"
            self.headers = {"Authorization": f"Bearer {self._v1_id_token()}"}
            self.daily_path = "/prices/daily_quotes"
            self.calendar_path = "/markets/trading_calendar"

    def _v1_id_token(self) -> str:
        refresh = os.environ.get("JQUANTS_REFRESH_TOKEN")
        if not refresh:
            mail, pw = os.environ.get("JQUANTS_MAIL"), os.environ.get("JQUANTS_PASSWORD")
            if not (mail and pw):
                raise SystemExit("J-Quants の認証情報が環境変数にありません")
            req = urllib.request.Request(
                "https://api.jquants.com/v1/token/auth_user",
                data=json.dumps({"mailaddress": mail, "password": pw}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                refresh = json.loads(r.read())["refreshToken"]
        q = urllib.parse.urlencode({"refreshtoken": refresh})
        req = urllib.request.Request(
            f"https://api.jquants.com/v1/token/auth_refresh?{q}", method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["idToken"]

    def _get_all(self, name: str, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 0
        while True:
            url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
            data = self.rec.get(f"{name}_p{page}", url, self.headers)
            rows += _rows(data)
            key = data.get("pagination_key") if isinstance(data, dict) else None
            if not key:
                return rows
            params = {**params, "pagination_key": key}
            page += 1

    def daily(self, name: str, code: str, start: str, end: str) -> list[dict[str, Any]]:
        return self._get_all(name, self.daily_path, {"code": code, "from": start, "to": end})

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        return self._get_all("jp_calendar", self.calendar_path, {"from": start, "to": end})


# --- Alpaca -----------------------------------------------------------------


class Alpaca:
    DATA = "https://data.alpaca.markets/v2"
    TRADING = "https://paper-api.alpaca.markets/v2"

    def __init__(self, rec: Recorder) -> None:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not (key and secret):
            raise SystemExit("Alpaca の認証情報が環境変数にありません")
        self.rec = rec
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def bars(
        self, name: str, symbols: list[str], start: str, end: str, adjustment: str
    ) -> dict[str, list[dict[str, Any]]]:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start,
            "end": end,
            "adjustment": adjustment,
            "feed": "iex",
            "limit": "10000",
        }
        out: dict[str, list[dict[str, Any]]] = {s: [] for s in symbols}
        page = 0
        while True:
            url = f"{self.DATA}/stocks/bars?{urllib.parse.urlencode(params)}"
            data = self.rec.get(f"{name}_p{page}", url, self.headers)
            for sym, bars in (data.get("bars") or {}).items():
                out.setdefault(sym, []).extend(bars)
            token = data.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token
            page += 1

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        url = f"{self.TRADING}/calendar?{urllib.parse.urlencode({'start': start, 'end': end})}"
        return _rows(self.rec.get("us_calendar", url, self.headers))


# --- 調査 -------------------------------------------------------------------


def probe_jp(rec: Recorder, today: dt.date) -> dict[str, Any]:
    jq = JQuants(rec)
    start = (today - dt.timedelta(days=LOOKBACK_CALENDAR_DAYS)).isoformat()
    end = today.isoformat()
    result: dict[str, Any] = {"api_version": jq.version, "lookback": {}, "split": {}}

    for code in JP_SYMBOLS:
        n_before = len(rec.requests)
        t0 = time.monotonic()
        rows = jq.daily(f"jp_{code}_lookback", code, start, end)
        prof = _profile(rows)
        prof["elapsed_sec"] = round(time.monotonic() - t0, 3)
        prof["requests"] = len(rec.requests) - n_before
        # 分割・併合で調整がかかった行（v1 は AdjustmentFactor）
        prof["adjustment_factor_not_1"] = [
            {"Date": r.get("Date"), "AdjustmentFactor": r.get("AdjustmentFactor")}
            for r in rows
            if r.get("AdjustmentFactor") not in (None, 1, 1.0)
        ]
        prof["zero_or_null_volume_days"] = [
            r.get("Date") for r in rows if not r.get("Volume") and not r.get("AdjustmentVolume")
        ]
        result["lookback"][code] = prof

    for code, s, e in SPLIT_WINDOWS["JP"]:
        rows = jq.daily(f"jp_{code}_split", code, s, e)
        result["split"][code] = [
            {
                k: r.get(k)
                for k in ("Date", "Close", "AdjustmentClose", "AdjustmentFactor", "Volume")
            }
            for r in rows
        ]

    far = jq.daily("jp_7203_far_past", "7203", FAR_PAST, end)
    result["earliest_available"] = _profile(far).get("first_date")

    cal = jq.calendar(start, (today + dt.timedelta(days=60)).isoformat())
    result["calendar"] = {
        "profile": _profile(cal),
        "non_business_days_sample": [r for r in cal if str(r.get("HolidayDivision")) != "1"][:15],
    }
    return result


def probe_jp_today(rec: Recorder, today: dt.date) -> dict[str, Any]:
    jq = JQuants(rec)
    rows = jq.daily("jp_7203_today", "7203", today.isoformat(), today.isoformat())
    return {
        "checked_at_jst": dt.datetime.now(JST).isoformat(timespec="seconds"),
        "target_date": today.isoformat(),
        "available": bool(rows),
        "row": rows[0] if rows else None,
    }


def probe_us(rec: Recorder, today: dt.date) -> dict[str, Any]:
    al = Alpaca(rec)
    start = (today - dt.timedelta(days=LOOKBACK_CALENDAR_DAYS)).isoformat()
    end = today.isoformat()
    result: dict[str, Any] = {"lookback": {}, "split": {}}

    n_before = len(rec.requests)
    t0 = time.monotonic()
    bars = al.bars("us_lookback", US_SYMBOLS, start, end, "all")
    result["lookback_batch"] = {
        "symbols": len(US_SYMBOLS),
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "requests": len(rec.requests) - n_before,
    }
    for sym, rows in bars.items():
        prof = _profile(rows)
        prof["zero_volume_days"] = [r.get("t") for r in rows if not r.get("v")]
        result["lookback"][sym] = prof

    for sym, s, e in SPLIT_WINDOWS["US"]:
        raw = al.bars(f"us_{sym}_split_raw", [sym], s, e, "raw")[sym]
        adj = al.bars(f"us_{sym}_split_all", [sym], s, e, "all")[sym]
        result["split"][sym] = [
            {"t": r.get("t"), "close_raw": r.get("c"), "close_all": a.get("c"), "v_raw": r.get("v")}
            for r, a in zip(raw, adj, strict=False)
        ]

    far = al.bars("us_SPY_far_past", ["SPY"], FAR_PAST, end, "all")["SPY"]
    result["earliest_available"] = _profile(far).get("first_date")

    cal = al.calendar(start, (today + dt.timedelta(days=60)).isoformat())
    result["calendar"] = {"profile": _profile(cal), "sample": cal[:5]}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["all", "jp", "us", "jp-today"])
    parser.add_argument("--date", type=dt.date.fromisoformat, default=dt.datetime.now(JST).date())
    args = parser.parse_args()

    stamp = dt.datetime.now(JST).strftime("%Y%m%d-%H%M%S")
    rec = Recorder(OUT_ROOT / f"{stamp}_{args.target}")
    summary: dict[str, Any] = {"target": args.target, "date": args.date.isoformat()}

    if args.target == "jp-today":
        summary["jp_today"] = probe_jp_today(rec, args.date)
    if args.target in ("all", "jp"):
        summary["jp"] = probe_jp(rec, args.date)
    if args.target in ("all", "us"):
        summary["us"] = probe_us(rec, args.date)
    summary["requests"] = rec.requests

    path = rec.out_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
