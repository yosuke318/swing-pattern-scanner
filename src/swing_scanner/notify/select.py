"""通知対象の抽出（基本設計書 5.2 の手順9）。

シグナル: スコアがパターンのしきい値以上、レジームが対象、除外されていない、出口を算出できた、
  クールダウン期間外。スコア順に並べる。
惜しかった銘柄: しきい値未満で near_miss.min_score 以上、レジームが対象、除外されていない。
  スコア順に near_miss.max_items 件まで。
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

from swing_scanner.config import Config, Market, Pattern
from swing_scanner.filters import FilterOutcome
from swing_scanner.notify.cooldown import in_cooldown
from swing_scanner.rules.engine import Evaluation


@dataclass(frozen=True)
class Judgement:
    """1銘柄 × 1パターンの判定結果（ルールエンジンと除外フィルタの結果をまとめたもの）。"""

    market: Market
    symbol: str
    name: str
    pattern: Pattern
    evaluation: Evaluation
    outcome: FilterOutcome

    @property
    def eligible(self) -> bool:
        """レジームが対象で、除外されていない。"""
        return not self.evaluation.skip_reason and not self.outcome.is_excluded


@dataclass(frozen=True)
class Selection:
    signals: list[Judgement]
    near_misses: list[Judgement]


def _by_score(items: list[Judgement]) -> list[Judgement]:
    return sorted(items, key=lambda j: (-j.evaluation.score, j.symbol, j.pattern.pattern_id))


def select(
    judgements: Sequence[Judgement],
    config: Config,
    notified: dict[str, dt.date],
    today: dt.date,
    sessions: Sequence[dt.date] | None = None,
    resend_same_day: bool = False,
) -> Selection:
    settings = config.settings.notification
    signals: list[Judgement] = []
    near: list[Judgement] = []
    for j in judgements:
        if not j.eligible:
            continue
        threshold = config.score_threshold(j.pattern)
        score = j.evaluation.score
        if score >= threshold:
            cooling = in_cooldown(
                notified,
                j.symbol,
                j.pattern.pattern_id,
                today,
                settings.cooldown_business_days,
                sessions,
                resend_same_day,
            )
            if j.evaluation.exit is not None and not cooling:
                signals.append(j)
        elif score >= settings.near_miss.min_score:
            near.append(j)
    return Selection(_by_score(signals), _by_score(near)[: settings.near_miss.max_items])
