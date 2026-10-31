"""config/ 配下の設定ファイルを読み込み、スキーマ検証する。

不正な設定は起動時に ConfigError で止める。必須項目の欠落、型違い、未知のキーはすべてエラー。
"""

from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from swing_scanner.rules.keys import KEYS

Market = Literal["JP", "US"]
VolRegime = Literal["low", "normal", "high"]
TrendRegime = Literal["uptrend", "range", "downtrend"]
TrendMethod = Literal["adx", "structure", "ma", "vote"]

PositiveInt = Annotated[StrictInt, Field(gt=0)]
PositiveFloat = Annotated[StrictFloat | StrictInt, Field(gt=0)]
Number = StrictFloat | StrictInt


class ConfigError(Exception):
    """設定ファイルの読み込み・検証の失敗。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# 日本株は4桁の銘柄コード（2024年以降の英字入りコード "130A" なども含む）。
# J-Quants の5桁コードとの変換は取得層で吸収する。米国株は Alpaca のティッカー（"BRK.B" など）
_SYMBOL_FORMATS: dict[str, re.Pattern[str]] = {
    "JP": re.compile(r"^[0-9][0-9A-Z][0-9][0-9A-Z]$"),
    "US": re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$"),
}


def _check_symbol_format(market: str, symbol: str) -> None:
    if not _SYMBOL_FORMATS[market].match(symbol):
        raise ValueError(f"{market} の銘柄コードとして不正です: {symbol!r}")


# --- settings.yaml ---------------------------------------------------------


class FetchSettings(_Model):
    lookback_days: PositiveInt


class MarketSettings(_Model):
    min_trading_value_20d: Annotated[Number, Field(ge=0)]
    # 市場環境フィルタと相対強度に使う指数ETF。監視銘柄とは別枠で、通知対象にはしない
    benchmark_symbol: StrictStr


class ScoringSettings(_Model):
    default_score_threshold: Annotated[Number, Field(ge=0, le=100)]


class ChannelSettings(_Model):
    enabled: StrictBool


class NotificationSettings(_Model):
    cooldown_business_days: Annotated[StrictInt, Field(ge=0)]
    email: ChannelSettings
    line: ChannelSettings


class FilterSettings(_Model):
    """除外フィルタ（基本設計書 4.7）の ON/OFF。市場環境は market_filter.mode で切り替える。"""

    liquidity: StrictBool
    earnings: StrictBool
    # 決算日の登録漏れの警告。直近この日数以内（未来を含む）に決算日の登録が無い監視銘柄を警告する
    earnings_registration_lookback_days: PositiveInt


class MarketFilterSettings(_Model):
    mode: Literal["off", "annotate", "suppress"]


class FeatureSettings(_Model):
    """指標計算のパラメータ（features/）。日次処理とバックテストで同じ値を使う。"""

    # ゴールデンクロスを判定する MA の組 [短期, 長期]
    golden_cross_pair: tuple[PositiveInt, PositiveInt]
    # スイング検出の前後の日数 N
    swing_window: PositiveInt
    # HH/HL 判定で比べるスイングの個数
    structure_pairs: Annotated[StrictInt, Field(ge=2)]
    trend_method: TrendMethod
    # 区分が連続でこの日数続くまで切り替えない。1 なら遅らせない
    hysteresis_days: PositiveInt

    @model_validator(mode="after")
    def _short_before_long(self) -> FeatureSettings:
        short, long = self.golden_cross_pair
        if short >= long:
            raise ValueError(f"golden_cross_pair は [短期, 長期] で書いてください: {[short, long]}")
        return self


class Settings(_Model):
    fetch: FetchSettings
    features: FeatureSettings
    markets: dict[Market, MarketSettings]
    scoring: ScoringSettings
    notification: NotificationSettings
    filters: FilterSettings
    market_filter: MarketFilterSettings

    @model_validator(mode="after")
    def _all_markets_present(self) -> Settings:
        missing = {"JP", "US"} - set(self.markets)
        if missing:
            raise ValueError(f"markets に {sorted(missing)} の設定がありません")
        for market, m in self.markets.items():
            _check_symbol_format(market, m.benchmark_symbol)
        return self


# --- symbols.yaml ----------------------------------------------------------


class Symbol(_Model):
    symbol: StrictStr
    name: StrictStr
    market: Market
    watched: StrictBool

    @model_validator(mode="after")
    def _symbol_format(self) -> Symbol:
        _check_symbol_format(self.market, self.symbol)
        return self


# --- events.yaml -----------------------------------------------------------


class Event(_Model):
    symbol: StrictStr
    market: Market
    date: dt.date
    type: Literal["earnings", "other"]
    note: StrictStr | None = None


# --- patterns/<pattern_id>/v<n>.yaml ---------------------------------------

Op = Literal["<", "<=", ">", ">=", "==", "!=", "between"]


class Condition(_Model):
    key: StrictStr
    op: Op
    value: StrictBool | Number | list[Number]
    weight: Annotated[Number, Field(gt=0)]

    @model_validator(mode="after")
    def _value_matches_op(self) -> Condition:
        spec = KEYS.get(self.key)
        if spec is None:
            raise ValueError(f"未知の key です: {self.key!r}（使える key は README の一覧を参照）")
        if spec.type == "bool" and not isinstance(self.value, bool):
            raise ValueError(
                f"{self.key} は真偽値の key です。value は true / false で書いてください"
            )
        if spec.type == "number" and isinstance(self.value, bool):
            raise ValueError(f"{self.key} は数値の key です。value に真偽値は使えません")
        if self.op == "between":
            if not isinstance(self.value, list) or len(self.value) != 2:
                raise ValueError(f"{self.key}: between の value は [下限, 上限] で書いてください")
            low, high = self.value
            if low > high:
                raise ValueError(f"{self.key}: between の下限が上限より大きいです")
        elif isinstance(self.value, list):
            raise ValueError(f"{self.key}: {self.op} の value にリストは使えません")
        elif isinstance(self.value, bool) and self.op not in ("==", "!="):
            raise ValueError(f"{self.key}: 真偽値の比較には == か != を使ってください")
        return self


class Regimes(_Model):
    vol: Annotated[list[VolRegime], Field(min_length=1)]
    trend: Annotated[list[TrendRegime], Field(min_length=1)]


class StopRule(_Model):
    # swing_low_or_atr: 直近スイング安値割れとエントリー − k×ATR のうち浅い方
    type: Literal["swing_low_or_atr", "atr"]
    atr_mult: PositiveFloat


class TargetRule(_Model):
    atr_mult: PositiveFloat


class ExitRule(_Model):
    stop: StopRule
    target: TargetRule
    max_holding_days: PositiveInt
    # 寄り付きが前日終値 + g×ATR 以上ならエントリーを見送る。省略時は見送らない
    skip_gap_atr_mult: PositiveFloat | None = None


class Pattern(_Model):
    pattern_id: Annotated[StrictStr, Field(pattern=r"^[a-z0-9_]+$")]
    version: PositiveInt
    name: StrictStr
    category: Literal["trend", "breakout", "pullback", "squeeze"]
    active: StrictBool
    regimes: Regimes
    conditions: Annotated[list[Condition], Field(min_length=1)]
    score_threshold: Annotated[Number, Field(ge=0, le=100)] | None = None
    exit: ExitRule

    @model_validator(mode="after")
    def _weights_sum_to_100(self) -> Pattern:
        total = sum(c.weight for c in self.conditions)
        if abs(total - 100) > 1e-9:
            raise ValueError(f"conditions の weight の合計が 100 ではありません（{total}）")
        keys = [c.key for c in self.conditions]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise ValueError(f"conditions の key が重複しています: {dup}")
        return self


# --- まとめて読む -----------------------------------------------------------


class Config(_Model):
    settings: Settings
    symbols: list[Symbol]
    events: list[Event]
    patterns: list[Pattern]

    def watched_symbols(self, market: Market) -> list[Symbol]:
        return [s for s in self.symbols if s.market == market and s.watched]

    def active_patterns(self) -> list[Pattern]:
        return [p for p in self.patterns if p.active]

    def pattern(self, pattern_id: str, version: int | None = None) -> Pattern:
        """pattern_id のパターンを返す。version を省略すると active なバージョン。

        バックテストで過去のバージョンを指定して読み込むときに使う。
        """
        for p in self.patterns:
            if p.pattern_id == pattern_id and (
                p.version == version or (version is None and p.active)
            ):
                return p
        which = f"v{version}" if version is not None else "active なバージョン"
        raise ConfigError(f"パターン {pattern_id} の {which} がありません")

    def score_threshold(self, pattern: Pattern) -> float:
        if pattern.score_threshold is not None:
            return pattern.score_threshold
        return self.settings.scoring.default_score_threshold


_PATTERN_FILE = re.compile(r"^v([1-9][0-9]*)\.yaml$")


def _read_yaml(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"{path}: ファイルがありません") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: YAML として読めません: {e}") from e


def _validate[T: BaseModel](model: type[T], data: Any, path: Path) -> T:
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path}: {e}") from e


def _validate_list[T: BaseModel](model: type[T], data: Any, path: Path) -> list[T]:
    if data is None:
        data = []
    if not isinstance(data, list):
        raise ConfigError(f"{path}: トップレベルはリストで書いてください")
    return [_validate(model, item, path) for item in data]


def load_patterns(patterns_dir: Path) -> list[Pattern]:
    patterns: list[Pattern] = []
    for path in sorted(patterns_dir.glob("*/*.yaml")):
        m = _PATTERN_FILE.match(path.name)
        if not m:
            raise ConfigError(f"{path}: ファイル名は v<番号>.yaml にしてください")
        pattern = _validate(Pattern, _read_yaml(path), path)
        if pattern.pattern_id != path.parent.name:
            raise ConfigError(
                f"{path}: pattern_id（{pattern.pattern_id}）がディレクトリ名と一致しません"
            )
        if pattern.version != int(m.group(1)):
            raise ConfigError(f"{path}: version（{pattern.version}）がファイル名と一致しません")
        patterns.append(pattern)

    active: dict[str, list[int]] = defaultdict(list)
    for p in patterns:
        if p.active:
            active[p.pattern_id].append(p.version)
    for pattern_id, versions in active.items():
        if len(versions) > 1:
            raise ConfigError(
                f"{patterns_dir / pattern_id}: active: true のバージョンが複数あります: {versions}"
            )
    return patterns


def load_config(config_dir: Path) -> Config:
    settings_path = config_dir / "settings.yaml"
    settings = _validate(Settings, _read_yaml(settings_path), settings_path)
    symbols_path = config_dir / "symbols.yaml"
    symbols = _validate_list(Symbol, _read_yaml(symbols_path), symbols_path)
    events_path = config_dir / "events.yaml"
    events = _validate_list(Event, _read_yaml(events_path), events_path)
    patterns = load_patterns(config_dir / "patterns")

    seen: set[tuple[str, str]] = set()
    for s in symbols:
        key = (s.market, s.symbol)
        if key in seen:
            raise ConfigError(f"{symbols_path}: 銘柄が重複しています: {s.market} {s.symbol}")
        seen.add(key)
        if s.watched and s.symbol == settings.markets[s.market].benchmark_symbol:
            raise ConfigError(
                f"{symbols_path}: {s.market} {s.symbol} はベンチマークです。"
                "settings.yaml の benchmark_symbol で持ち、監視銘柄には入れないでください"
            )
    for e in events:
        if (e.market, e.symbol) not in seen:
            raise ConfigError(
                f"{events_path}: symbols.yaml に無い銘柄のイベントです: {e.market} {e.symbol}"
            )

    return Config(settings=settings, symbols=symbols, events=events, patterns=patterns)
