"""ルールエンジン（基本設計書 4.3 / 4.6、要件 5.3）。日次処理とバックテストで同じ関数を使う。

パターン定義の条件セットと1日分の指標（compute_features の1行）を照合し、真偽値だけでなく
条件ごとの合否・実測値・閾値とスコアを返す。全条件 AND だけにすると、4条件中3つ合致のような
境界の銘柄が見えなくなるため。

- 条件の状態は passed / failed / unevaluable の3つ。指標が欠けている（期間不足・指数欠損）ときは
  unevaluable にし、合致扱いにも不成立扱いにもしない（スコアには入らず、matched にもならない）
- between は両端を含む（下限 <= 値 <= 上限）
- スコア = passed の条件の weight の合計（100点満点）。matched = 全条件 passed かつレジームが対象
- パターンの regimes に当日の vol_regime / trend_regime が入っていなければ対象外（skip_reason に
  理由を入れ、matched にしない）。レジームが評価不能のときも対象外。条件の評価結果は判定ログの
  ために残す
- 出口は exit_plan で算出する。日次では当日終値を基準にした目安で、バックテストは実際の
  エントリー価格を渡して計算し直す
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from swing_scanner.config import Condition, ExitRule, Pattern
from swing_scanner.rules.keys import KEYS, scalar

Status = Literal["passed", "failed", "unevaluable"]


@dataclass(frozen=True)
class ConditionResult:
    key: str
    op: str
    threshold: float | bool | list[float]
    actual: float | bool | None
    status: Status
    weight: float

    def to_log(self) -> dict[str, Any]:
        """判定ログ（state/results/ の breakdown）に入れる形。passed は評価不能なら null。"""
        actual = self.actual
        if isinstance(actual, float) and math.isinf(actual):
            actual = "inf" if actual > 0 else "-inf"  # JSON に Infinity を書かない
        return {
            "key": self.key,
            "op": self.op,
            "threshold": self.threshold,
            "actual": actual,
            "passed": None if self.status == "unevaluable" else self.status == "passed",
            "weight": self.weight,
        }


@dataclass(frozen=True)
class ExitPlan:
    base_price: float
    stop: float
    # 損切りをどちらで決めたか。swing_low: 直近スイング安値、atr: 基準価格 − k×ATR
    stop_basis: Literal["swing_low", "atr"]
    target: float
    max_holding_days: int
    # 寄り付きがこの価格以上の窓開けならエントリーを見送る（skip_gap_atr_mult が無ければ None）
    skip_if_open_at_or_above: float | None


@dataclass(frozen=True)
class Evaluation:
    pattern_id: str
    version: int
    conditions: tuple[ConditionResult, ...]
    score: float
    matched: bool
    # レジームが対象外・評価不能なときの理由。対象なら空文字
    skip_reason: str
    # 終値か ATR14 が欠けていれば None
    exit: ExitPlan | None

    @property
    def unevaluable_keys(self) -> list[str]:
        return [c.key for c in self.conditions if c.status == "unevaluable"]

    def breakdown(self) -> list[dict[str, Any]]:
        return [c.to_log() for c in self.conditions]


def compare(op: str, actual: float | bool, threshold: float | bool | list[float]) -> bool:
    if op == "between":
        low, high = threshold  # type: ignore[misc]
        return low <= actual <= high
    ops = {
        "<": lambda a, t: a < t,
        "<=": lambda a, t: a <= t,
        ">": lambda a, t: a > t,
        ">=": lambda a, t: a >= t,
        "==": lambda a, t: a == t,
        "!=": lambda a, t: a != t,
    }
    return ops[op](actual, threshold)


def evaluate_condition(condition: Condition, row: Mapping[str, Any]) -> ConditionResult:
    actual = KEYS[condition.key].value(row)
    if actual is None or (isinstance(actual, float) and math.isnan(actual)):
        status: Status = "unevaluable"
    else:
        status = "passed" if compare(condition.op, actual, condition.value) else "failed"
    return ConditionResult(
        key=condition.key,
        op=condition.op,
        threshold=condition.value,
        actual=actual,
        status=status,
        weight=float(condition.weight),
    )


def regime_skip_reason(pattern: Pattern, row: Mapping[str, Any]) -> str:
    targets = {"vol_regime": pattern.regimes.vol, "trend_regime": pattern.regimes.trend}
    for column, allowed in targets.items():
        value = scalar(row.get(column))
        if value is None:
            return f"{column} が評価不能"
        if value not in allowed:
            return f"{column}={value}（対象: {', '.join(allowed)}）"
    return ""


def exit_plan(
    rule: ExitRule, base_price: Any, atr14: Any, swing_low: Any = None
) -> ExitPlan | None:
    """出口を算出する（基本設計書 4.6）。

    損切り: swing_low_or_atr は「直近スイング安値」と「基準価格 − k×ATR」のうち浅い方（高い方）。
      スイング安値が無い、または基準価格以上（すでに割り込んでいる）なら ATR の方を使う。
      atr は常に 基準価格 − k×ATR。
    利確: 基準価格 + m×ATR。
    """
    base, atr, swing = scalar(base_price), scalar(atr14), scalar(swing_low)
    if base is None or atr is None:
        return None
    stop = base - rule.stop.atr_mult * atr
    basis: Literal["swing_low", "atr"] = "atr"
    if rule.stop.type == "swing_low_or_atr" and swing is not None and stop < swing < base:
        stop, basis = swing, "swing_low"
    gap = rule.skip_gap_atr_mult
    return ExitPlan(
        base_price=base,
        stop=stop,
        stop_basis=basis,
        target=base + rule.target.atr_mult * atr,
        max_holding_days=rule.max_holding_days,
        skip_if_open_at_or_above=None if gap is None else base + gap * atr,
    )


def evaluate(pattern: Pattern, row: Mapping[str, Any]) -> Evaluation:
    """1日分の指標（compute_features の1行。辞書でもよい）をパターンと照合する。"""
    results = tuple(evaluate_condition(c, row) for c in pattern.conditions)
    skip_reason = regime_skip_reason(pattern, row)
    return Evaluation(
        pattern_id=pattern.pattern_id,
        version=pattern.version,
        conditions=results,
        score=sum(r.weight for r in results if r.status == "passed"),
        matched=not skip_reason and all(r.status == "passed" for r in results),
        skip_reason=skip_reason,
        exit=exit_plan(
            pattern.exit, row.get("close"), row.get("atr14"), row.get("swing_low_price")
        ),
    )


def evaluate_all(patterns: list[Pattern], row: Mapping[str, Any]) -> list[Evaluation]:
    return [evaluate(p, row) for p in patterns]
