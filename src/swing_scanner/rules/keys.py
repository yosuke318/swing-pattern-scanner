"""パターン条件の key として使える値の一覧（[3-1] の key 一覧、[3-2] の入力仕様）。

指標の列（features.compute_features の出力）をそのまま使う key と、列から導出する派生 key がある。
パターン定義の読み込み時に、ここに無い key と、型の合わない value（真偽値の key に数値など）を弾く。

key は一度使ったら消さない・意味を変えない。過去のバージョンのパターン定義が読めなくなるため。
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

KeyType = Literal["number", "bool"]


def scalar(value: Any) -> Any:
    """行の値を Python の値にする。NaN / NA / None は None（評価不能）。"""
    if value is None:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


@dataclass(frozen=True)
class KeySpec:
    name: str
    type: KeyType
    columns: tuple[str, ...]
    description: str
    # 列の値から key の値を作る。None なら1つ目の列の値をそのまま使う
    derive: Callable[..., Any] | None = None
    # 通知文での表示名。None なら description
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.description

    def value(self, row: Mapping[str, Any]) -> float | bool | None:
        """行から値を取り出す。元の列のどれかが欠けていれば None（評価不能）。"""
        values = [scalar(row.get(c)) for c in self.columns]
        if any(v is None for v in values):
            return None
        v = self.derive(*values) if self.derive else values[0]
        return bool(v) if self.type == "bool" else float(v)


def _column(name: str, type_: KeyType, description: str) -> KeySpec:
    return KeySpec(name, type_, (name,), description)


_SPECS = [
    # --- 価格・トレンド
    _column("close", "number", "終値"),
    *(_column(f"ma{n}", "number", f"{n}日単純移動平均") for n in (5, 25, 75, 200)),
    *(
        _column(f"close_minus_ma{n}_atr_ratio", "number", f"(終値 − MA{n}) ÷ ATR14")
        for n in (5, 25, 75, 200)
    ),
    KeySpec(
        "golden_cross_within_days",
        "number",
        ("days_since_golden_cross",),
        "直近のゴールデンクロスからの営業日数（当日 0、未発生は inf）。`<=` で「X日以内」",
        label="ゴールデンクロスからの営業日数",
    ),
    _column("golden_cross_flag", "bool", "当日ゴールデンクロスが発生した"),
    _column("adx14", "number", "ADX14（トレンドの強さ）"),
    _column("plus_di", "number", "+DI"),
    _column("minus_di", "number", "-DI"),
    KeySpec(
        "plus_di_above_minus_di",
        "bool",
        ("plus_di", "minus_di"),
        "+DI > -DI（上向きのトレンド）",
        lambda plus, minus: plus > minus,
    ),
    # --- スイング・構造
    _column("swing_low_higher_than_prior", "bool", "直近の確定スイング安値 > その前の安値"),
    KeySpec(
        "swing_high_higher_than_prior",
        "bool",
        ("swing_high_price", "prior_swing_high_price"),
        "直近の確定スイング高値 > その前の高値",
        lambda latest, prior: latest > prior,
    ),
    KeySpec(
        "close_above_swing_high",
        "bool",
        ("close", "swing_high_price"),
        "終値 > 直近の確定スイング高値（高値の更新）",
        lambda close, high: close > high,
    ),
    KeySpec(
        "hh_hl_uptrend",
        "bool",
        ("hh_hl_structure",),
        "hh_hl_structure が uptrend",
        lambda structure: structure == "uptrend",
    ),
    # --- ボラティリティ
    _column("atr14", "number", "ATR14（Wilder）"),
    _column("atr14_pct", "number", "ATR14 ÷ 終値（比率）"),
    _column("atr252_percentile", "number", "atr14_pct の直近252日での順位（0〜100）"),
    # --- 出来高・相対強度・流動性
    _column("volume_ratio_20d", "number", "当日出来高 ÷ 前日までの20日平均出来高"),
    _column("relative_strength_20d", "number", "20日リターンの指数ETFとの差（比率）"),
    _column("trading_value_20d", "number", "20日平均売買代金"),
]

KEYS: dict[str, KeySpec] = {spec.name: spec for spec in _SPECS}
