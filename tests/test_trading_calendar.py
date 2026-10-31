import csv
import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from swing_scanner.daily import start_run
from swing_scanner.sources import jquants
from swing_scanner.sources.alpaca import AlpacaClient
from swing_scanner.sources.http import ApiError
from swing_scanner.sources.jquants import JQuantsClient
from swing_scanner.trading_calendar import (
    AlpacaCalendar,
    CalendarError,
    JQuantsCalendar,
    Session,
    parse_alpaca_calendar,
    parse_jquants_calendar,
    resolve_target,
)

JST = ZoneInfo("Asia/Tokyo")
NY = ZoneInfo("America/New_York")
D = dt.date


class FakeCalendar:
    """土日と holidays を休場日とするカレンダー。引けは JP 15:30 / US 16:00。"""

    def __init__(self, market: str, holidays: set[dt.date] = frozenset()) -> None:
        self.tz, self.close = {"JP": (JST, dt.time(15, 30)), "US": (NY, dt.time(16))}[market]
        self.holidays = holidays
        self.calls: list[tuple[dt.date, dt.date]] = []

    def sessions(self, start: dt.date, end: dt.date) -> dict[dt.date, Session]:
        self.calls.append((start, end))
        out = {}
        day = start
        while day <= end:
            if day.weekday() < 5 and day not in self.holidays:
                out[day] = Session(day, dt.datetime.combine(day, self.close, self.tz))
            day += dt.timedelta(days=1)
        return out


# 2026-11-03（火）文化の日。2026-11-26（木）は米国の感謝祭
JP_HOLIDAYS = {D(2026, 11, 3), D(2026, 11, 23)}
US_HOLIDAYS = {D(2026, 11, 26)}


# --- 対象日の決定 -------------------------------------------------------------


def test_jp_trading_day_after_close() -> None:
    now = dt.datetime(2026, 11, 2, 18, 0, tzinfo=JST)
    target = resolve_target("JP", FakeCalendar("JP", JP_HOLIDAYS), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 2), True)


def test_jp_holiday_is_skipped_not_moved_back() -> None:
    now = dt.datetime(2026, 11, 3, 18, 0, tzinfo=JST)
    target = resolve_target("JP", FakeCalendar("JP", JP_HOLIDAYS), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 3), False)


def test_jp_weekend_is_skipped() -> None:
    now = dt.datetime(2026, 11, 7, 18, 0, tzinfo=JST)  # 土曜
    assert not resolve_target("JP", FakeCalendar("JP"), now).is_trading_day


def test_us_runs_on_japanese_holiday_with_ny_date() -> None:
    # 日本時間 11/3（文化の日）の朝 = NY 時間 11/2（月）の夕方。米国株は通常どおり判定する
    now = dt.datetime(2026, 11, 3, 7, 0, tzinfo=JST)
    target = resolve_target("US", FakeCalendar("US", US_HOLIDAYS), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 2), True)


def test_us_target_is_ny_date_not_jst_date() -> None:
    # 日本時間の土曜の朝 = NY 時間の金曜の夕方
    now = dt.datetime(2026, 11, 7, 7, 0, tzinfo=JST)
    target = resolve_target("US", FakeCalendar("US"), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 6), True)


def test_us_holiday_is_skipped() -> None:
    now = dt.datetime(2026, 11, 27, 7, 0, tzinfo=JST)  # NY 11/26 17:00（感謝祭）
    target = resolve_target("US", FakeCalendar("US", US_HOLIDAYS), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 26), False)


def test_before_close_targets_previous_day() -> None:
    now = dt.datetime(2026, 11, 4, 10, 0, tzinfo=NY)
    target = resolve_target("US", FakeCalendar("US"), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 3), True)


def test_before_close_previous_day_holiday_is_skipped() -> None:
    now = dt.datetime(2026, 11, 4, 10, 0, tzinfo=JST)  # 前日 11/3 は文化の日
    target = resolve_target("JP", FakeCalendar("JP", JP_HOLIDAYS), now)
    assert (target.date, target.is_trading_day) == (D(2026, 11, 3), False)


def test_override_date_is_used_as_is() -> None:
    now = dt.datetime(2026, 11, 4, 10, 0, tzinfo=JST)  # 引け前でも繰り下げない
    cal = FakeCalendar("JP", JP_HOLIDAYS)
    assert resolve_target("JP", cal, now, override=D(2026, 11, 4)).date == D(2026, 11, 4)
    holiday = resolve_target("JP", cal, now, override=D(2026, 11, 23))
    assert (holiday.date, holiday.is_trading_day) == (D(2026, 11, 23), False)


def test_calendar_is_fetched_with_lookback_window() -> None:
    cal = FakeCalendar("JP")
    resolve_target("JP", cal, dt.datetime(2026, 11, 2, 18, 0, tzinfo=JST))
    assert cal.calls == [(D(2026, 10, 23), D(2026, 11, 2))]


# --- 実行の先頭（runs.csv への記録） --------------------------------------------


def _runs(state_dir) -> list[dict[str, str]]:
    path = state_dir / "runs.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_start_run_records_skipped_holiday(tmp_path) -> None:
    now = dt.datetime(2026, 11, 3, 9, 0, tzinfo=dt.UTC)  # JST 18:00
    target = start_run("JP", FakeCalendar("JP", JP_HOLIDAYS), tmp_path, now, "abc123")
    assert not target.is_trading_day
    [row] = _runs(tmp_path)
    assert row["market"] == "JP"
    assert row["target_date"] == "2026-11-03"
    assert row["status"] == "skipped_holiday"
    assert row["commit_sha"] == "abc123"


def test_start_run_on_trading_day_records_nothing(tmp_path) -> None:
    now = dt.datetime(2026, 11, 2, 9, 0, tzinfo=dt.UTC)
    assert start_run("JP", FakeCalendar("JP"), tmp_path, now, "abc123").is_trading_day
    assert _runs(tmp_path) == []


def test_start_run_propagates_calendar_error(tmp_path) -> None:
    class Broken:
        def sessions(self, start, end):
            raise CalendarError("down")

    with pytest.raises(CalendarError):
        start_run("US", Broken(), tmp_path, dt.datetime.now(dt.UTC), "abc123")
    assert _runs(tmp_path) == []


# --- J-Quants のカレンダー -----------------------------------------------------


def _jq_rows(start: dt.date, divisions: str, key: str = "HolidayDivision") -> list[dict]:
    return [
        {"Date": (start + dt.timedelta(days=i)).isoformat(), key: div}
        for i, div in enumerate(divisions)
    ]


def test_parse_jquants_calendar() -> None:
    rows = _jq_rows(D(2026, 11, 2), "1023")
    sessions = parse_jquants_calendar(rows, D(2026, 11, 2), D(2026, 11, 5))
    assert sorted(sessions) == [D(2026, 11, 2), D(2026, 11, 4)]
    assert sessions[D(2026, 11, 2)].close == dt.datetime(2026, 11, 2, 15, 30, tzinfo=JST)
    assert sessions[D(2026, 11, 4)].close == dt.datetime(2026, 11, 4, 11, 30, tzinfo=JST)


def test_parse_jquants_calendar_accepts_v2_field_name() -> None:
    rows = _jq_rows(D(2026, 11, 2), "10", key="HolDiv")
    assert list(parse_jquants_calendar(rows, D(2026, 11, 2), D(2026, 11, 3))) == [D(2026, 11, 2)]


def test_parse_jquants_calendar_rejects_missing_dates() -> None:
    rows = _jq_rows(D(2026, 11, 2), "11")
    with pytest.raises(CalendarError, match="2026-11-04"):
        parse_jquants_calendar(rows, D(2026, 11, 2), D(2026, 11, 4))


def test_parse_jquants_calendar_rejects_unknown_division() -> None:
    with pytest.raises(CalendarError):
        parse_jquants_calendar(_jq_rows(D(2026, 11, 2), "9"), D(2026, 11, 2), D(2026, 11, 2))


def test_jquants_calendar_uses_version_specific_path() -> None:
    class FakeClient:
        version = "v1"

        def __init__(self) -> None:
            self.calls: list = []

        def get_all(self, path, params, rows_keys):
            self.calls.append((path, params))
            return _jq_rows(D(2026, 11, 2), "1")

    client = FakeClient()
    JQuantsCalendar(client).sessions(D(2026, 11, 2), D(2026, 11, 2))
    assert client.calls == [
        ("/markets/trading_calendar", {"from": "2026-11-02", "to": "2026-11-02"})
    ]


def test_jquants_calendar_wraps_api_error() -> None:
    class FailingClient:
        version = "v2"

        def get_all(self, path, params, rows_keys):
            raise ApiError("HTTP 500")

    with pytest.raises(CalendarError, match="HTTP 500"):
        JQuantsCalendar(FailingClient()).sessions(D(2026, 11, 2), D(2026, 11, 2))


# --- Alpaca のカレンダー -------------------------------------------------------


def test_parse_alpaca_calendar() -> None:
    rows = [
        {"date": "2026-11-25", "open": "09:30", "close": "16:00"},
        {"date": "2026-11-27", "open": "09:30", "close": "13:00"},  # 感謝祭の翌日は短縮取引
    ]
    sessions = parse_alpaca_calendar(rows, D(2026, 11, 25), D(2026, 11, 27))
    assert sorted(sessions) == [D(2026, 11, 25), D(2026, 11, 27)]
    assert sessions[D(2026, 11, 27)].close == dt.datetime(2026, 11, 27, 13, 0, tzinfo=NY)


def test_parse_alpaca_calendar_rejects_empty() -> None:
    with pytest.raises(CalendarError):
        parse_alpaca_calendar([], D(2026, 11, 20), D(2026, 11, 30))


def test_alpaca_calendar_requests_range(monkeypatch) -> None:
    seen = {}

    def fake_request_json(url, headers=None, **kw):
        seen["url"], seen["headers"] = url, headers
        return [{"date": "2026-11-02", "open": "09:30", "close": "16:00"}]

    monkeypatch.setattr("swing_scanner.sources.alpaca.request_json", fake_request_json)
    client = AlpacaClient.from_env({"APCA_API_KEY_ID": "id", "APCA_API_SECRET_KEY": "secret"})
    AlpacaCalendar(client).sessions(D(2026, 10, 23), D(2026, 11, 2))
    assert seen["url"] == (
        "https://paper-api.alpaca.markets/v2/calendar?start=2026-10-23&end=2026-11-02"
    )
    assert seen["headers"] == {"APCA-API-KEY-ID": "id", "APCA-API-SECRET-KEY": "secret"}


def test_alpaca_requires_credentials() -> None:
    with pytest.raises(ApiError):
        AlpacaClient.from_env({})


# --- J-Quants のクライアント ---------------------------------------------------


def test_jquants_v2_uses_api_key() -> None:
    client = JQuantsClient.from_env({"JQUANTS_API_KEY": "k"})
    assert (client.version, client.headers) == ("v2", {"x-api-key": "k"})


def test_jquants_v1_fetches_id_token_every_time(monkeypatch) -> None:
    calls = []

    def fake_request_json(url, method="GET", body=None, **kw):
        calls.append((method, url.split("?")[0], body))
        if url.endswith("/token/auth_user"):
            return {"refreshToken": "r"}
        return {"idToken": "id-token"}

    monkeypatch.setattr(jquants, "request_json", fake_request_json)
    env = {"JQUANTS_MAIL": "m@example.com", "JQUANTS_PASSWORD": "p"}
    client = JQuantsClient.from_env(env)
    assert client.headers == {"Authorization": "Bearer id-token"}
    assert calls == [
        (
            "POST",
            "https://api.jquants.com/v1/token/auth_user",
            {"mailaddress": "m@example.com", "password": "p"},
        ),
        ("POST", "https://api.jquants.com/v1/token/auth_refresh", None),
    ]


def test_jquants_requires_credentials() -> None:
    with pytest.raises(ApiError):
        JQuantsClient.from_env({})


def test_jquants_get_all_follows_pagination(monkeypatch) -> None:
    pages = [
        {"trading_calendar": [{"Date": "2026-11-02"}], "pagination_key": "next"},
        {"trading_calendar": [{"Date": "2026-11-03"}]},
    ]
    urls = []

    def fake_request_json(url, headers=None, **kw):
        urls.append(url)
        return pages[len(urls) - 1]

    monkeypatch.setattr(jquants, "request_json", fake_request_json)
    client = JQuantsClient("v1", {"Authorization": "Bearer t"})
    rows = client.get_all("/markets/trading_calendar", {"from": "a"}, ("trading_calendar",))
    assert [r["Date"] for r in rows] == ["2026-11-02", "2026-11-03"]
    assert urls[1].endswith("from=a&pagination_key=next")


def test_jquants_get_all_rejects_unexpected_shape(monkeypatch) -> None:
    monkeypatch.setattr(jquants, "request_json", lambda url, headers=None, **kw: {"message": "x"})
    client = JQuantsClient("v2", {"x-api-key": "k"})
    with pytest.raises(ApiError, match="trading_calendar/data"):
        client.get_all("/markets/calendar", {}, ("trading_calendar", "data"))
