"""日足の取得元（[1-4] J-Quants、[1-5] Alpaca）。応答の形は公式ドキュメントに沿った作り物。"""

import datetime as dt
import math
from pathlib import Path

import pandas as pd
import pytest

from swing_scanner.sources import jquants
from swing_scanner.sources.alpaca import AlpacaClient, AlpacaDailySource, parse_bars
from swing_scanner.sources.base import CsvSource, SourceUnavailable, source_for
from swing_scanner.sources.http import ApiError
from swing_scanner.sources.jquants import (
    JQuantsClient,
    JQuantsDailySource,
    parse_daily_quotes,
    to_jquants_code,
)
from swing_scanner.validation import validate_ohlcv

D = dt.date


def v2_row(date: str, close: float, adj: float = 1.0, **kw) -> dict:
    """v2 の株価四本値の1行。調整前は adj で割り戻した値にする。"""
    raw = close / adj
    row = {
        "Date": date,
        "Code": "67580",
        "O": raw,
        "H": raw * 1.01,
        "L": raw * 0.99,
        "C": raw,
        "Vo": 1000 * adj,
        "Va": raw * 1000 * adj,
        "AdjFactor": 1.0,
        "AdjO": close,
        "AdjH": close * 1.01,
        "AdjL": close * 0.99,
        "AdjC": close,
        "AdjVo": 1000.0,
        "UL": "0",
        "LL": "0",
    }
    return row | kw


# --- J-Quants ----------------------------------------------------------------------


def test_to_jquants_code() -> None:
    assert to_jquants_code("7203") == "72030"
    assert to_jquants_code("130A") == "130A0"
    assert to_jquants_code("72030") == "72030"


def test_parse_v2_uses_adjusted_prices() -> None:
    # ソニーの 1:5 分割（2024-10-01）。調整前は 14,000 → 2,850 と段差があり、調整後は連続する
    rows = [
        v2_row("2024-10-01", 2850.0, AdjFactor=0.2),
        v2_row("2024-09-30", 2800.0, adj=0.2),
    ]
    df = parse_daily_quotes(rows, "v2")
    assert list(df.columns[:7]) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
    ]
    assert df["date"].tolist() == [D(2024, 9, 30), D(2024, 10, 1)]
    assert df["close"].tolist() == [2800.0, 2850.0]
    assert df["volume"].tolist() == [1000.0, 1000.0]
    # 売買代金は調整の影響を受けないので調整前の Va をそのまま使う
    assert df["trading_value"].iloc[0] == pytest.approx(14000 * 200)


def test_parse_v2_keeps_no_trade_days_and_limit_flags() -> None:
    rows = [
        v2_row("2024-10-01", 2850.0, UL="1"),
        v2_row("2024-10-02", 0.0)
        | {k: None for k in ("AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "Va")},
    ]
    df = parse_daily_quotes(rows, "v2")
    assert df["upper_limit"].tolist() == [True, False]
    assert df["lower_limit"].tolist() == [False, False]
    assert math.isnan(df["close"].iloc[1]) and math.isnan(df["volume"].iloc[1])


def test_parse_v1_columns() -> None:
    rows = [
        {
            "Date": "2024-10-01",
            "AdjustmentOpen": 1.0,
            "AdjustmentHigh": 2.0,
            "AdjustmentLow": 0.5,
            "AdjustmentClose": 1.5,
            "AdjustmentVolume": 100.0,
            "TurnoverValue": 150.0,
        }
    ]
    df = parse_daily_quotes(rows, "v1")
    assert df.iloc[0][["open", "high", "low", "close", "volume", "trading_value"]].tolist() == [
        1.0,
        2.0,
        0.5,
        1.5,
        100.0,
        150.0,
    ]
    assert "upper_limit" not in df


def test_parse_rejects_broken_rows_and_handles_empty() -> None:
    with pytest.raises(ApiError):
        parse_daily_quotes([{"AdjC": 1}], "v2")
    assert parse_daily_quotes([], "v2").empty


class FakeJQuantsClient:
    version = "v2"

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    def get_all(self, path, params, rows_keys):
        self.calls.append((path, params, rows_keys))
        return self.rows


def test_jquants_source_fetches_by_symbol_with_from_to() -> None:
    client = FakeJQuantsClient([v2_row("2024-10-01", 2850.0)])
    source = JQuantsDailySource(lambda: client)
    df = source.fetch("6758", D(2024, 1, 1), D(2024, 10, 1))
    assert client.calls == [
        (
            "/equities/bars/daily",
            {"code": "67580", "from": "2024-01-01", "to": "2024-10-01"},
            ("daily_quotes", "data"),
        )
    ]
    assert validate_ohlcv(df, D(2024, 10, 1)) is None


def test_jquants_source_authenticates_lazily() -> None:
    calls = []

    def factory():
        calls.append(1)
        raise ApiError("J-Quants の認証情報が環境変数にありません")

    source = JQuantsDailySource(factory)
    assert calls == []  # 作っただけでは認証しない（休場日に API を叩かない）
    with pytest.raises(SourceUnavailable, match="認証"):
        source.fetch("7203", D(2024, 1, 1), D(2024, 10, 1))


def test_jquants_client_throttles_requests(monkeypatch) -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(sec):
        sleeps.append(sec)
        now[0] += sec

    monkeypatch.setattr(jquants, "request_json", lambda url, headers=None: {"data": []})
    client = JQuantsClient("v2", {"x-api-key": "k"}, clock=lambda: now[0], sleep=sleep)
    client.get_all("/equities/bars/daily", {}, ("data",))
    now[0] += 0.3
    client.get_all("/equities/bars/daily", {}, ("data",))
    now[0] += 2.0
    client.get_all("/equities/bars/daily", {}, ("data",))
    assert sleeps == [pytest.approx(0.8)]  # 2回目だけ 1.1 秒の間隔まで待つ


# --- Alpaca ------------------------------------------------------------------------


def bar(t: str, close: float, **kw) -> dict:
    return {
        "t": t,
        "o": close,
        "h": close + 1,
        "l": close - 1,
        "c": close,
        "v": 1000,
        "vw": close + 0.5,
    } | kw


def test_parse_bars_uses_ny_date_and_vwap() -> None:
    df = parse_bars([bar("2024-06-11T04:00:00Z", 121.0), bar("2024-06-10T04:00:00Z", 120.0)])
    assert df["date"].tolist() == [D(2024, 6, 10), D(2024, 6, 11)]
    assert df["trading_value"].iloc[0] == pytest.approx(120.5 * 1000)
    no_vwap = parse_bars([bar("2024-06-10T04:00:00Z", 120.0, vw=None)])
    assert no_vwap["trading_value"].iloc[0] == pytest.approx(120.0 * 1000)
    with pytest.raises(ApiError):
        parse_bars([{"t": "2024-06-10T04:00:00Z"}])


class FakeAlpacaClient:
    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages
        self.calls: list[tuple] = []

    def get_data(self, path, params):
        self.calls.append((path, dict(params)))
        return self.pages[len(self.calls) - 1]


def test_alpaca_source_requests_split_adjusted_sip(monkeypatch) -> None:
    monkeypatch.delenv("APCA_DATA_FEED", raising=False)
    client = FakeAlpacaClient(
        [
            {"bars": {"NVDA": [bar("2024-06-10T04:00:00Z", 120.0)]}, "next_page_token": "p2"},
            {
                "bars": {
                    "NVDA": [bar("2024-06-11T04:00:00Z", 121.0), bar("2024-06-12T04:00:00Z", 125.0)]
                }
            },
        ]
    )
    df = AlpacaDailySource(lambda: client).fetch("NVDA", D(2024, 6, 1), D(2024, 6, 11))
    path, params = client.calls[0]
    assert path == "/stocks/bars"
    assert params == {
        "symbols": "NVDA",
        "timeframe": "1Day",
        "start": "2024-06-01",
        "end": "2024-06-12",
        "adjustment": "split",
        "feed": "sip",
        "limit": "10000",
    }
    assert client.calls[1][1]["page_token"] == "p2"
    # end の翌日まで取り、対象日より後の足は落とす
    assert df["date"].tolist() == [D(2024, 6, 10), D(2024, 6, 11)]
    assert validate_ohlcv(df, D(2024, 6, 11)) is None


def test_alpaca_feed_can_be_switched(monkeypatch) -> None:
    monkeypatch.setenv("APCA_DATA_FEED", "iex")
    client = FakeAlpacaClient([{"bars": {}}])
    df = AlpacaDailySource(lambda: client).fetch("NVDA", D(2024, 6, 1), D(2024, 6, 11))
    assert client.calls[0][1]["feed"] == "iex"
    assert df.empty


def test_alpaca_source_requires_credentials() -> None:
    def factory():
        return AlpacaClient.from_env({})

    with pytest.raises(SourceUnavailable, match="APCA_API_KEY_ID"):
        AlpacaDailySource(factory).fetch("NVDA", D(2024, 6, 1), D(2024, 6, 11))


# --- 切り替え ----------------------------------------------------------------------


def test_source_for_market(tmp_path: Path) -> None:
    assert isinstance(source_for("JP"), JQuantsDailySource)
    assert isinstance(source_for("US"), AlpacaDailySource)
    assert isinstance(source_for("US", tmp_path), CsvSource)


def test_csv_source(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2024-06-10", "2024-06-11"],
            "open": [1, 2],
            "high": [2, 3],
            "low": [0.5, 1],
            "close": [1.5, 2],
            "volume": [10, 20],
        }
    ).to_csv(tmp_path / "NVDA.csv", index=False)
    df = CsvSource(tmp_path).fetch("NVDA", D(2024, 6, 11), D(2024, 6, 30))
    assert df["date"].tolist() == [D(2024, 6, 11)]
