"""取得データの検証とリトライ（[1-6]）。"""

import datetime as dt
import io
import math
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from swing_scanner.config import load_config
from swing_scanner.features import compute_features
from swing_scanner.sources import http
from swing_scanner.sources.http import ApiError, request_json
from swing_scanner.validation import validate_ohlcv

ROOT = Path(__file__).resolve().parents[1]
DAY = dt.date(2026, 10, 30)


def frame(n: int = 10, end: dt.date = DAY, **columns) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=n).date
    close = [100.0 + i for i in range(n)]
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [c + 2 for c in close],
            "low": [c - 2 for c in close],
            "close": close,
            "volume": [1000.0] * n,
        }
    )
    for name, values in columns.items():
        df[name] = values
    return df


def set_row(df: pd.DataFrame, i: int, **values) -> pd.DataFrame:
    df = df.copy()
    for k, v in values.items():
        df.loc[df.index[i], k] = v
    return df


# --- 判定から外すもの ---------------------------------------------------------------


def test_clean_data_is_ok() -> None:
    result = validate_ohlcv(frame(), DAY)
    assert (result.status, result.reason, result.warnings) == ("ok", "", [])
    assert len(result.data) == 10


@pytest.mark.parametrize(
    "df, status, reason",
    [
        (frame().drop(columns="volume"), "missing", "列がありません"),
        (frame().iloc[0:0], "missing", "データがありません"),
        (
            frame(end=dt.date(2026, 10, 29)),
            "missing",
            "2026-10-30 のデータがありません（最終 2026-10-29。データ未更新）",
        ),
        (frame().iloc[::-1], "anomaly", "日付の並びが不正"),
        (set_row(frame(), -1, volume=0.0), "missing", "2026-10-30 に売買がありません"),
        (
            set_row(frame(), -1, open=None, high=None, low=None, close=None, volume=None),
            "missing",
            "売買がありません",
        ),
        (set_row(frame(), 3, volume=None), "missing", "一部が欠けています"),
        (set_row(frame(), 3, high=90.0), "anomaly", "矛盾"),  # 高値 < 安値・始値・終値
        (set_row(frame(), 3, low=110.0), "anomaly", "矛盾"),  # 安値 > 始値・終値
        (set_row(frame(), 3, close=-1.0), "anomaly", "矛盾"),
        (set_row(frame(), 3, volume=-5.0), "anomaly", "矛盾"),
    ],
)
def test_excluded(df, status, reason) -> None:
    result = validate_ohlcv(df, DAY)
    assert result.status == status
    assert reason in result.reason
    assert result.data is None and not result.ok


def test_contradiction_message_names_the_day() -> None:
    result = validate_ohlcv(set_row(frame(), 3, high=90.0), DAY)
    assert str(frame()["date"].iloc[3]) in result.reason


# --- 警告に留めるもの ---------------------------------------------------------------


def test_no_trade_days_are_dropped_with_warning() -> None:
    df = set_row(frame(), 4, open=None, high=None, low=None, close=None, volume=None)
    result = validate_ohlcv(df, DAY)
    assert result.ok
    assert len(result.data) == 9
    assert result.data["close"].notna().all()
    assert result.warnings == [f"売買なしの日が1日あります（{df['date'].iloc[4]}。除いて計算）"]


def test_gaps_against_trading_calendar() -> None:
    df = frame()
    sessions = list(df["date"])
    gapped = df.drop(index=[2, 5]).reset_index(drop=True)
    result = validate_ohlcv(gapped, DAY, sessions)
    assert result.ok
    assert result.warnings == [f"立会日の行が2日ありません（{sessions[2]}, {sessions[5]}）"]
    # 取得期間より前の立会日は数えない
    assert validate_ohlcv(df.iloc[3:].reset_index(drop=True), DAY, sessions).warnings == []


def test_past_zero_volume_is_a_warning() -> None:
    result = validate_ohlcv(set_row(frame(), 2, volume=0.0), DAY)
    assert result.ok and "出来高 0 の日が1日" in result.warnings[0]


def test_jump_is_a_warning_unless_limit_move() -> None:
    # 105 → 160 は +52%（翌日の 160 → 106 は -34% で閾値内）
    jumped = set_row(frame(), 5, open=160.0, high=162.0, low=158.0, close=160.0)
    result = validate_ohlcv(jumped, DAY)
    assert result.ok
    assert result.warnings == [
        f"前日比 ±35% を超える値動きが1日あります（{jumped['date'].iloc[5]}。調整漏れの可能性）"
    ]
    limit = jumped.assign(upper_limit=[i == 5 for i in range(10)], lower_limit=[False] * 10)
    assert validate_ohlcv(limit, DAY).warnings == []


def test_up_to_three_dates_are_listed() -> None:
    df = frame(20)
    for i in (2, 4, 6, 8, 10):
        df = set_row(df, i, volume=0.0)
    warning = validate_ohlcv(df, DAY).warnings[0]
    assert warning.startswith("出来高 0 の日が5日あります（") and warning.endswith(" など）")


# --- リトライ ----------------------------------------------------------------------


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"oops"))


def scripted(monkeypatch, outcomes):
    calls = []

    def urlopen(req, timeout):
        calls.append(req.get_method())
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    monkeypatch.setattr(http.urllib.request, "urlopen", urlopen)
    return calls


def test_get_retries_transient_errors(monkeypatch) -> None:
    calls = scripted(monkeypatch, [http_error(503), urllib.error.URLError("timeout"), b'{"ok": 1}'])
    sleeps: list[float] = []
    assert request_json("https://x", sleep=sleeps.append) == {"ok": 1}
    assert len(calls) == 3 and sleeps == [2.0, 4.0]


def test_rate_limit_waits_longer(monkeypatch) -> None:
    scripted(monkeypatch, [http_error(429), b"{}"])
    sleeps: list[float] = []
    request_json("https://x", sleep=sleeps.append)
    assert sleeps == [10.0]


def test_gives_up_after_retries(monkeypatch) -> None:
    calls = scripted(monkeypatch, [http_error(500)] * 3)
    with pytest.raises(ApiError, match="HTTP 500"):
        request_json("https://x", sleep=lambda s: None)
    assert len(calls) == 3


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(monkeypatch, code) -> None:
    calls = scripted(monkeypatch, [http_error(code), b"{}"])
    with pytest.raises(ApiError, match=f"HTTP {code}"):
        request_json("https://x", sleep=lambda s: None)
    assert len(calls) == 1


def test_post_is_not_retried(monkeypatch) -> None:
    calls = scripted(monkeypatch, [http_error(503), b"{}"])
    with pytest.raises(ApiError):
        request_json("https://x", method="POST", body={}, sleep=lambda s: None)
    assert calls == ["POST"]


def test_error_message_hides_query(monkeypatch) -> None:
    scripted(monkeypatch, [http_error(401)])
    with pytest.raises(ApiError) as e:
        request_json("https://x/token?refreshtoken=SECRET", sleep=lambda s: None)
    assert "SECRET" not in str(e.value)


def test_nan_free_after_validation_feeds_features() -> None:
    # 売買なしの日を除けば指標が途切れない（Wilder 平滑は NaN が入るとそれ以降がすべて NaN）
    df = frame(300)
    df = set_row(df, 150, open=None, high=None, low=None, close=None, volume=None)
    result = validate_ohlcv(df, DAY)
    settings = load_config(ROOT / "config").settings.features
    features = compute_features(result.data, None, settings)
    assert not math.isnan(features["atr14"].iloc[-1])
    assert not math.isnan(features["adx14"].iloc[-1])
