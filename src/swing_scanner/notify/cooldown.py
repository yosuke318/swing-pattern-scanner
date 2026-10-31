"""クールダウン（state/notified.json）。同一銘柄・同一パターンは一定の営業日数だけ再送しない。

notified.json は "<symbol>|<pattern_id>" → 最終通知日（判定対象日）。最終通知日から数えて
cooldown_business_days 営業日が経つまでは再送しない（例: 5 なら、月曜に送った銘柄は翌週月曜から
再送できる）。営業日は取引カレンダーの立会日で数え、カレンダーが無い期間は土日だけを休みとする。
"""

import datetime as dt
from collections.abc import Iterable, Sequence

import numpy as np

from swing_scanner.state import notified_key


def business_days_between(
    last: dt.date, today: dt.date, sessions: Sequence[dt.date] | None = None
) -> int:
    """last の翌日から today まで（today を含む）の営業日数。"""
    if sessions and min(sessions) <= last:
        return sum(1 for d in sessions if last < d <= today)
    return int(np.busday_count(last + dt.timedelta(days=1), today + dt.timedelta(days=1)))


def in_cooldown(
    notified: dict[str, dt.date],
    symbol: str,
    pattern_id: str,
    today: dt.date,
    days: int,
    sessions: Sequence[dt.date] | None = None,
    resend_same_day: bool = False,
) -> bool:
    """クールダウン中か。resend_same_day なら、同じ対象日に送ったものは再送してよい（--force）。"""
    last = notified.get(notified_key(symbol, pattern_id))
    if last is None:
        return False
    if resend_same_day and last == today:
        return False
    return business_days_between(last, today, sessions) < days


def record_notified(
    notified: dict[str, dt.date], sent: Iterable[tuple[str, str]], today: dt.date
) -> dict[str, dt.date]:
    """送った (symbol, pattern_id) の最終通知日を today にした新しい辞書を返す。"""
    return notified | {notified_key(symbol, pattern_id): today for symbol, pattern_id in sent}


def prune(
    notified: dict[str, dt.date],
    today: dt.date,
    days: int,
    sessions: Sequence[dt.date] | None = None,
) -> dict[str, dt.date]:
    """クールダウンを過ぎたエントリを消した新しい辞書を返す。"""
    return {
        key: last
        for key, last in notified.items()
        if business_days_between(last, today, sessions) < days
    }
