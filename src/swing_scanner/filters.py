"""除外フィルタ（基本設計書 4.7、要件 5.4）。判定の後、通知の前に挟む。

除外は「通知しない」だけで、判定自体は判定ログに残す（理由は excluded_reason）。

- 流動性: trading_value_20d が市場別の下限未満なら除外。20日分そろわず値が無いときも除外する
  （上場直後など、流動性を確かめられない銘柄に通知しない）
- 決算またぎ: 判定日から max_holding_days 営業日先までに決算日があれば除外。判定日当日の決算も
  含める（引け後の発表で翌日の寄り付きが動くため）。パターンごとに max_holding_days が違うので、
  パターン × 銘柄で判定する
- 市場環境: 指数ETFの trend_regime が downtrend なら、mode に従って注記（annotate）か
  除外（suppress）。指数の trend_regime が分からない日は除外せず、注記だけ付ける

各フィルタは settings.yaml の filters / market_filter で個別に止められる。
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from swing_scanner.config import Config, Event, Market, Pattern
from swing_scanner.rules.keys import scalar


@dataclass(frozen=True)
class FilterOutcome:
    # 除外の理由。空なら除外しない
    excluded: tuple[str, ...]
    # 除外はしないが通知に添える注記（市場環境の annotate など）
    notes: tuple[str, ...]

    @property
    def is_excluded(self) -> bool:
        return bool(self.excluded)


def business_day_offset(
    start: dt.date, days: int, sessions: Sequence[dt.date] | None = None
) -> dt.date:
    """start から days 営業日先の日付。

    sessions（取引カレンダーの立会日。start 以降を十分先まで含む昇順）があればそれで数える。
    無ければ土日だけを休みとして数える（祝日の分だけ窓が短くなるので、日次処理では
    sessions を渡す）。
    """
    if sessions is not None:
        ahead = [d for d in sessions if d > start]
        if len(ahead) < days:
            raise ValueError(f"取引カレンダーが {start} から {days} 営業日先まで足りません")
        return ahead[days - 1] if days > 0 else start
    return np.busday_offset(start, days, roll="forward").astype(dt.date)


def liquidity_reason(trading_value_20d: Any, minimum: float) -> str | None:
    value = scalar(trading_value_20d)
    if value is None:
        return "流動性: 20日平均売買代金が評価不能"
    if value < minimum:
        return f"流動性: 20日平均売買代金 {value:,.0f} < 下限 {minimum:,.0f}"
    return None


def earnings_reason(
    events: Sequence[Event],
    market: Market,
    symbol: str,
    date: dt.date,
    horizon_end: dt.date,
) -> str | None:
    for e in sorted(events, key=lambda e: e.date):
        mine = e.type == "earnings" and e.market == market and e.symbol == symbol
        if mine and date <= e.date <= horizon_end:
            return f"決算: {e.date} に決算（保有想定期間 {date}〜{horizon_end}）"
    return None


def market_condition(benchmark_trend: Any, mode: str) -> tuple[str | None, str | None]:
    """(除外の理由, 注記) を返す。"""
    if mode == "off":
        return None, None
    trend = scalar(benchmark_trend)
    if trend is None:
        return None, "市場環境: 指数の trend_regime が判定不能"
    if trend != "downtrend":
        return None, None
    message = "市場環境: 指数が downtrend"
    return (message, None) if mode == "suppress" else (None, message)


def apply_filters(
    config: Config,
    market: Market,
    symbol: str,
    date: dt.date,
    pattern: Pattern,
    row: Any,
    benchmark_trend: Any,
    sessions: Sequence[dt.date] | None = None,
) -> FilterOutcome:
    """1銘柄 × 1パターンに除外フィルタを当てる。row は compute_features の判定日の行。"""
    settings = config.settings
    excluded: list[str] = []
    notes: list[str] = []
    if settings.filters.liquidity:
        minimum = settings.markets[market].min_trading_value_20d
        reason = liquidity_reason(row.get("trading_value_20d"), minimum)
        if reason:
            excluded.append(reason)
    if settings.filters.earnings:
        horizon_end = business_day_offset(date, pattern.exit.max_holding_days, sessions)
        reason = earnings_reason(config.events, market, symbol, date, horizon_end)
        if reason:
            excluded.append(reason)
    reason, note = market_condition(benchmark_trend, settings.market_filter.mode)
    if reason:
        excluded.append(reason)
    if note:
        notes.append(note)
    return FilterOutcome(tuple(excluded), tuple(notes))


def missing_earnings_warnings(config: Config, market: Market, today: dt.date) -> list[str]:
    """決算日の登録漏れの警告。直近 N 日以内（未来を含む）に決算日の登録が無い監視銘柄を挙げる。

    決算フィルタを止めているときは警告しない。
    """
    if not config.settings.filters.earnings:
        return []
    lookback = config.settings.filters.earnings_registration_lookback_days
    since = today - dt.timedelta(days=lookback)
    registered = {
        e.symbol
        for e in config.events
        if e.type == "earnings" and e.market == market and e.date >= since
    }
    return [
        f"{s.symbol} {s.name}: 直近{lookback}日以内の決算日が events.yaml に登録されていません"
        for s in config.watched_symbols(market)
        if s.symbol not in registered
    ]
