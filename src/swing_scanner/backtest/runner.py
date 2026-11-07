"""バックテストの実行（[4-1]）。過去データに、日次処理と同じ関数を当て直す。

  指標計算: features.compute_features（[2-7]）
  判定:     rules.engine.evaluate（[3-2]）と filters.apply_filters（[3-4]）
  通知対象: レジームが対象・必須条件を満たす・スコアがしきい値以上・除外されていない・クールダウン外
            （日次処理で通知される条件と同じ）
  トレード: backtest.trades.simulate_trade（[4-2]）。同じ銘柄 × パターンで保有中なら新規に入らない

指標は期間全体をまとめて計算する（各行の値はその日までのデータだけで決まることをテスト済み）。
評価期間の前に、指標が出そろうまでの助走期間（fetch.lookback_days 営業日相当）を取る。
"""

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from swing_scanner.backtest.cache import OhlcvCache
from swing_scanner.backtest.trades import Trade, simulate_trade
from swing_scanner.config import Config, Market, Pattern, Symbol
from swing_scanner.features import compute_features
from swing_scanner.filters import apply_filters
from swing_scanner.notify.cooldown import business_days_between
from swing_scanner.rules.engine import evaluate
from swing_scanner.rules.keys import scalar
from swing_scanner.sources.http import ApiError
from swing_scanner.validation import validate_ohlcv

CALENDAR_DAYS_PER_BUSINESS_DAY = 1.6
# 取引カレンダーの代わりに使う立会日（ベンチマークの日付）を、評価期間の後ろに延ばす日数
SESSIONS_EXTENSION_DAYS = 60


@dataclass(frozen=True)
class SignalRecord:
    """通知対象になったシグナル（日次処理なら通知されていたもの）と、トレードにしたかどうか。"""

    symbol: str
    date: dt.date
    score: float
    # トレードにしなかった理由。"" ならトレードにした
    # gap: 窓開けで見送り / holding: 保有中 / no_next_bar: 翌営業日のデータが無い / no_exit_plan
    skip_reason: str = ""


@dataclass(frozen=True)
class SymbolData:
    symbol: str
    rows: int
    fetched_at: dt.datetime | None
    from_cache: bool
    error: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    pattern: Pattern
    market: Market
    start: dt.date
    end: dt.date
    symbols: list[Symbol]
    cost_per_side: float
    threshold: float
    trades: list[Trade]
    signals: list[SignalRecord]
    data: list[SymbolData]
    benchmark: SymbolData
    # 除外フィルタで外したシグナル候補の数（理由の先頭 → 件数）
    excluded: dict[str, int]
    # クールダウン中で通知されなかった候補の数
    cooled_down: int


def _load(
    cache: OhlcvCache,
    market: Market,
    symbol: str,
    start: dt.date,
    end: dt.date,
    refresh: bool,
) -> tuple[pd.DataFrame | None, SymbolData]:
    try:
        entry = cache.load(market, symbol, start, end, refresh=refresh)
    except (ApiError, OSError, ValueError) as e:
        return None, SymbolData(symbol, 0, None, False, error=f"取得に失敗しました: {e}")
    df = entry.data
    if df.empty:
        return None, SymbolData(symbol, 0, entry.fetched_at, entry.from_cache, error="データなし")
    result = validate_ohlcv(df, df["date"].iloc[-1])
    info = SymbolData(
        symbol,
        len(df),
        entry.fetched_at,
        entry.from_cache,
        error="" if result.ok else result.reason,
        warnings=result.warnings,
    )
    return (result.data if result.ok else None), info


def _sessions(dates: Sequence[dt.date]) -> list[dt.date]:
    """ベンチマークの日付を立会日とみなし、後ろを平日で延ばす（決算またぎの窓を数えるため）。"""
    if not dates:
        return []
    last = dates[-1]
    extra = np.arange(
        np.datetime64(last + dt.timedelta(days=1)),
        np.datetime64(last + dt.timedelta(days=SESSIONS_EXTENSION_DAYS)),
    )
    ext = [d.astype(dt.date) for d in extra if np.is_busday(d)]
    return sorted(dates) + ext


@dataclass
class Prepared:
    """期間全体の指標と、レジーム・必須条件を満たした日（スコアはしきい値の前）。

    パターンの条件とレジームだけで決まる部分を先に計算しておき、しきい値や出口を変えた試行
    （ウォークフォワードの探索、[4-3]）では simulate だけを繰り返す。
    """

    market: Market
    symbols: list[Symbol]
    data: list[SymbolData]
    benchmark: SymbolData
    features: dict[str, pd.DataFrame]
    # 銘柄 → [(行番号, 日付, スコア)]
    candidates: dict[str, list[tuple[int, dt.date, float]]]
    bench_trend: dict[dt.date, object]
    sessions: list[dt.date]
    # (銘柄, 行番号, max_holding_days) → 除外の理由
    _filter_cache: dict[tuple[str, int, int], tuple[str, ...]] = field(default_factory=dict)


def prepare(
    config: Config,
    pattern: Pattern,
    market: Market,
    symbols: Sequence[Symbol],
    start: dt.date,
    end: dt.date,
    cache: OhlcvCache,
    refresh: bool = False,
) -> Prepared:
    """start〜end（と、その前の助走期間）のデータを読み、指標を計算し、判定の候補日を集める。"""
    settings = config.settings
    warmup = math.ceil(settings.fetch.lookback_days * CALENDAR_DAYS_PER_BUSINESS_DAY)
    fetch_start = start - dt.timedelta(days=warmup)

    bench_symbol = settings.markets[market].benchmark_symbol
    bench_df, bench_info = _load(cache, market, bench_symbol, fetch_start, end, refresh)
    bench_trend: dict[dt.date, object] = {}
    sessions: list[dt.date] = []
    if bench_df is not None:
        bench_features = compute_features(bench_df, None, settings.features)
        bench_trend = dict(zip(bench_features["date"], bench_features["trend_regime"], strict=True))
        sessions = _sessions(list(bench_df["date"]))

    data: list[SymbolData] = []
    features: dict[str, pd.DataFrame] = {}
    candidates: dict[str, list[tuple[int, dt.date, float]]] = {}
    for symbol in symbols:
        df, info = _load(cache, market, symbol.symbol, fetch_start, end, refresh)
        data.append(info)
        if df is None:
            continue
        f = compute_features(df, bench_df, settings.features)
        features[symbol.symbol] = f
        found = []
        for i, row in f.iterrows():
            if start <= row["date"] <= end:
                ev = evaluate(pattern, row)
                if not ev.skip_reason:
                    found.append((int(i), row["date"], ev.score))
        candidates[symbol.symbol] = found
    return Prepared(
        market, list(symbols), data, bench_info, features, candidates, bench_trend, sessions
    )


def simulate(
    config: Config,
    prepared: Prepared,
    pattern: Pattern,
    start: dt.date,
    end: dt.date,
    data_end: dt.date | None = None,
) -> BacktestResult:
    """start〜end のシグナルを通知対象の条件で選び、仮想トレードにする。

    pattern は prepare に渡したものと条件・レジームが同じで、しきい値や出口だけが違ってよい。
    data_end を渡すと、それより後の足を使わない（トレードは data_end の終値で閉じる）。
    ウォークフォワードの学習期間で、評価期間の値動きが探索に漏れないようにするため。
    """
    settings = config.settings
    market = prepared.market
    threshold = config.score_threshold(pattern)
    cost = settings.backtest.cost_per_side[market]
    cooldown_days = settings.notification.cooldown_business_days
    sessions = prepared.sessions or None

    trades: list[Trade] = []
    signals: list[SignalRecord] = []
    excluded: dict[str, int] = {}
    cooled_down = 0
    for symbol in prepared.symbols:
        features = prepared.features.get(symbol.symbol)
        if features is None:
            continue
        bars = features if data_end is None else features[features["date"] <= data_end]
        last_notified: dt.date | None = None
        open_until: dt.date | None = None
        for i, day, score in prepared.candidates[symbol.symbol]:
            if not (start <= day <= end) or score < threshold:
                continue
            row = features.iloc[i]
            key = (symbol.symbol, i, pattern.exit.max_holding_days)
            if key not in prepared._filter_cache:
                outcome = apply_filters(
                    config,
                    market,
                    symbol.symbol,
                    day,
                    pattern,
                    row,
                    prepared.bench_trend.get(day),
                    sessions,
                )
                prepared._filter_cache[key] = outcome.excluded
            reasons = prepared._filter_cache[key]
            if reasons:
                for reason in reasons:
                    k = reason.split(":")[0]
                    excluded[k] = excluded.get(k, 0) + 1
                continue
            if last_notified is not None and (
                business_days_between(last_notified, day, sessions) < cooldown_days
            ):
                cooled_down += 1
                continue
            last_notified = day
            if open_until is not None and day < open_until:
                signals.append(SignalRecord(symbol.symbol, day, score, "holding"))
                continue
            result = simulate_trade(
                bars,
                i,
                pattern.exit,
                float(row["atr14"]),
                scalar(row.get("swing_low_price")),
                cost,
                symbol.symbol,
                pattern.pattern_id,
                pattern.version,
            )
            if isinstance(result, str):
                signals.append(SignalRecord(symbol.symbol, day, score, result))
                continue
            signals.append(SignalRecord(symbol.symbol, day, score))
            trades.append(result)
            open_until = result.exit_date
    return BacktestResult(
        pattern=pattern,
        market=market,
        start=start,
        end=end,
        symbols=prepared.symbols,
        cost_per_side=cost,
        threshold=threshold,
        trades=trades,
        signals=signals,
        data=prepared.data,
        benchmark=prepared.benchmark,
        excluded=excluded,
        cooled_down=cooled_down,
    )


def run_backtest(
    config: Config,
    pattern: Pattern,
    market: Market,
    symbols: Sequence[Symbol],
    start: dt.date,
    end: dt.date,
    cache: OhlcvCache,
    refresh: bool = False,
) -> BacktestResult:
    prepared = prepare(config, pattern, market, symbols, start, end, cache, refresh)
    return simulate(config, prepared, pattern, start, end)
