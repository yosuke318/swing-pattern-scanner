"""取引カレンダーによる休場日判定と対象日の決定（基本設計書 5.2 の手順2）。

GitHub Actions の cron は取引所の祝日を知らないので、休場日にも起動する。休場日は正常終了として
扱い、[1-6] の異常検知が休場日のデータ欠損で鳴らないようにする。休場日リストは自前で持たない。

- 日本株: J-Quants の取引カレンダー
- 米国株: Alpaca の calendar エンドポイント

カレンダーを取得・解釈できなかったときは CalendarError で止める（異常終了）。休場日を営業日と
取り違えても、データ欠損として異常扱いになるだけで誤ったシグナルは出ないため、推測で続けない。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from swing_scanner.config import Market
from swing_scanner.sources.alpaca import AlpacaClient
from swing_scanner.sources.http import ApiError
from swing_scanner.sources.jquants import JQuantsClient

MARKET_TZ: dict[Market, ZoneInfo] = {
    "JP": ZoneInfo("Asia/Tokyo"),
    "US": ZoneInfo("America/New_York"),
}

# J-Quants の HolidayDivision（v2 は HolDiv）。立会がある区分には引け時刻を対応させる。
# J-Quants は引け時刻を返さないので固定値で持つ（2024-11-05 以降の大引けは 15:30）
JP_CLOSE_BY_DIVISION: dict[str, dt.time | None] = {
    "0": None,  # 非営業日
    "1": dt.time(15, 30),  # 営業日
    "2": dt.time(11, 30),  # 東証半日立会日
    "3": None,  # 非営業日（祝日取引あり。デリバティブのみで、株式の立会は無い）
}

# 対象日の前に遡って取るカレンダーの日数。前日への繰り下げと、取得結果の妥当性確認に使う
# （11日間ずっと立会が無いことはないので、立会日が1日も無ければ取得の失敗とみなす）
LOOKBACK_DAYS = 10


class CalendarError(Exception):
    """取引カレンダーを取得・解釈できなかった。"""


@dataclass(frozen=True)
class Session:
    """立会のある日と、その日の引け時刻（タイムゾーン付き）。"""

    date: dt.date
    close: dt.datetime


class TradingCalendar(Protocol):
    def sessions(self, start: dt.date, end: dt.date) -> dict[dt.date, Session]:
        """start〜end（両端を含む）の立会日を返す。休場日は含めない。"""
        ...


def _days(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    for i in range((end - start).days + 1):
        yield start + dt.timedelta(days=i)


def parse_jquants_calendar(
    rows: list[dict[str, Any]], start: dt.date, end: dt.date
) -> dict[dt.date, Session]:
    """J-Quants の取引カレンダーを解釈する。

    休場日も1日1行で返るので、start〜end の全日がそろっているか確かめる。
    """
    tz = MARKET_TZ["JP"]
    seen: set[dt.date] = set()
    sessions: dict[dt.date, Session] = {}
    for row in rows:
        try:
            day = dt.date.fromisoformat(row["Date"])
            division = str(row.get("HolidayDivision", row.get("HolDiv")))
            close = JP_CLOSE_BY_DIVISION[division]
        except (KeyError, TypeError, ValueError) as e:
            raise CalendarError(f"J-Quants の取引カレンダーを解釈できません: {row!r}") from e
        seen.add(day)
        if close is not None and start <= day <= end:
            sessions[day] = Session(day, dt.datetime.combine(day, close, tz))
    missing = [d for d in _days(start, end) if d not in seen]
    if missing:
        raise CalendarError(
            f"J-Quants の取引カレンダーに {missing[0]} などの日付がありません（{len(missing)}日分）"
        )
    return sessions


def parse_alpaca_calendar(
    rows: list[dict[str, Any]], start: dt.date, end: dt.date
) -> dict[dt.date, Session]:
    """Alpaca の calendar を解釈する。立会日だけが返り、時刻は米国東部時間。"""
    tz = MARKET_TZ["US"]
    sessions: dict[dt.date, Session] = {}
    for row in rows:
        try:
            day = dt.date.fromisoformat(row["date"])
            close = dt.time.fromisoformat(row["close"])
        except (KeyError, TypeError, ValueError) as e:
            raise CalendarError(f"Alpaca の calendar を解釈できません: {row!r}") from e
        if start <= day <= end:
            sessions[day] = Session(day, dt.datetime.combine(day, close, tz))
    if not sessions:
        raise CalendarError(f"Alpaca の calendar に {start}〜{end} の立会日がありません")
    return sessions


class JQuantsCalendar:
    PATHS = {"v1": "/markets/trading_calendar", "v2": "/markets/calendar"}
    ROWS_KEYS = ("trading_calendar", "data")

    def __init__(self, client: JQuantsClient) -> None:
        self.client = client

    def sessions(self, start: dt.date, end: dt.date) -> dict[dt.date, Session]:
        params = {"from": start.isoformat(), "to": end.isoformat()}
        try:
            rows = self.client.get_all(self.PATHS[self.client.version], params, self.ROWS_KEYS)
        except ApiError as e:
            raise CalendarError(f"J-Quants の取引カレンダーを取得できません: {e}") from e
        return parse_jquants_calendar(rows, start, end)


class AlpacaCalendar:
    def __init__(self, client: AlpacaClient) -> None:
        self.client = client

    def sessions(self, start: dt.date, end: dt.date) -> dict[dt.date, Session]:
        params = {"start": start.isoformat(), "end": end.isoformat()}
        try:
            rows = self.client.get_trading("/calendar", params)
        except ApiError as e:
            raise CalendarError(f"Alpaca の calendar を取得できません: {e}") from e
        if not isinstance(rows, list):
            raise CalendarError("Alpaca の calendar が想定外の応答です")
        return parse_alpaca_calendar(rows, start, end)


def calendar_from_env(market: Market) -> TradingCalendar:
    try:
        if market == "JP":
            return JQuantsCalendar(JQuantsClient.from_env())
        return AlpacaCalendar(AlpacaClient.from_env())
    except ApiError as e:
        raise CalendarError(str(e)) from e


@dataclass(frozen=True)
class TargetDay:
    market: Market
    date: dt.date
    is_trading_day: bool


def resolve_target(
    market: Market,
    calendar: TradingCalendar,
    now: dt.datetime,
    override: dt.date | None = None,
) -> TargetDay:
    """判定の対象日を決め、立会日かどうかを返す。

    対象日は「その市場の現地時刻で、直近に引けた日」。米国株は日本時間の翌朝に実行するので、
    実行日（日本時間）ではなく NY 時間の日付になる。現地の日付の立会がまだ引けていなければ
    前日にする。その日が休場日なら is_trading_day=False（前の立会日まで遡らない。遡ると
    前回と同じ日を二重に判定してしまう）。

    override は手動実行で対象日を指定したとき。引け前の繰り下げはせず、その日をそのまま使う。
    """
    local_now = now.astimezone(MARKET_TZ[market])
    target = override or local_now.date()
    sessions = calendar.sessions(target - dt.timedelta(days=LOOKBACK_DAYS), target)
    if override is None:
        session = sessions.get(target)
        if session is not None and local_now < session.close:
            target -= dt.timedelta(days=1)
    return TargetDay(market, target, target in sessions)


class WeekdayCalendar:
    """土日だけを休場とするカレンダー。API の認証情報なしで動作確認するとき用（祝日を知らない）。"""

    CLOSE: dict[Market, dt.time] = {"JP": dt.time(15, 30), "US": dt.time(16, 0)}

    def __init__(self, market: Market) -> None:
        self.market = market

    def sessions(self, start: dt.date, end: dt.date) -> dict[dt.date, Session]:
        tz, close = MARKET_TZ[self.market], self.CLOSE[self.market]
        return {
            d: Session(d, dt.datetime.combine(d, close, tz))
            for d in _days(start, end)
            if d.weekday() < 5
        }
