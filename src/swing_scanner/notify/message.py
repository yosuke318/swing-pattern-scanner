"""通知文の生成（要件 5.5）。1回の実行（市場ごとに1日1回）で1通にまとめる。

読んだ後にチャートを開いて確認し、翌朝に注文を出すかを決める前提で、「なぜ候補か」「どこで入るか」
「どこで諦めるか」「いつまで持つか」に絞る。出口は発注しないツールの目安なので断定しない。

メールは根拠まで含め、LINE は要点（銘柄・スコア・損切り・利確・保有日数）だけにする。
該当なしの日も、シグナル0件・警告の有無が分かる1通を作る。
"""

import datetime as dt
import math
from dataclasses import dataclass, field

from swing_scanner.config import Market
from swing_scanner.notify.select import Judgement
from swing_scanner.rules.engine import ConditionResult
from swing_scanner.rules.keys import KEYS

MARKET_NAMES: dict[Market, str] = {"JP": "日本株", "US": "米国株"}
WEEKDAYS = "月火水木金土日"
# LINE のテキストメッセージの上限（5,000文字）に余裕を持たせる
LINE_MAX_CHARS = 4500


@dataclass(frozen=True)
class Report:
    market: Market
    target_date: dt.date
    signals: list[Judgement]
    near_misses: list[Judgement] = field(default_factory=list)
    # 取得失敗・異常値・決算日の登録漏れなど
    warnings: list[str] = field(default_factory=list)
    # スコアのしきい値（銘柄ごとの表示用。pattern_id → しきい値）
    thresholds: dict[str, float] = field(default_factory=dict)


def format_price(value: float, market: Market) -> str:
    if market == "US":
        return f"{value:,.2f}"
    text = f"{value:,.1f}"
    return text[:-2] if text.endswith(".0") else text


def format_value(value: float | bool | None) -> str:
    if value is None:
        return "評価不能"
    if isinstance(value, bool):
        return "はい" if value else "いいえ"
    if math.isinf(value):
        return "なし"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    text = f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_threshold(c: ConditionResult) -> str:
    if c.op == "between":
        low, high = c.threshold  # type: ignore[misc]
        return f"{format_value(float(low))}〜{format_value(float(high))}"
    if isinstance(c.threshold, bool):
        expected = c.threshold if c.op == "==" else not c.threshold
        return format_value(expected)
    return f"{c.op} {format_value(c.threshold)}"  # type: ignore[arg-type]


def _pct(price: float, base: float) -> str:
    return f"{(price - base) / base * 100:+.1f}%"


def _score(j: Judgement, report: Report) -> str:
    threshold = report.thresholds.get(j.pattern.pattern_id)
    suffix = f"（しきい値 {format_value(threshold)}）" if threshold is not None else ""
    return f"スコア {format_value(j.evaluation.score)}/100{suffix}"


def subject(report: Report) -> str:
    head = f"【{report.market}】{report.target_date:%Y-%m-%d}"
    body = f"シグナル{len(report.signals)}件" if report.signals else "該当なし"
    if report.warnings:
        body += f" / 警告{len(report.warnings)}件"
    return f"{head} {body}"


def _condition_line(c: ConditionResult) -> str:
    mark = {"passed": "○", "failed": "×", "unevaluable": "－"}[c.status]
    name = KEYS[c.key].display_name
    return f"    {mark} {name}: {format_value(c.actual)}（基準 {format_threshold(c)}）"


def _signal_block(i: int, j: Judgement, report: Report) -> list[str]:
    plan = j.evaluation.exit
    assert plan is not None  # シグナルは出口を算出できたものだけ
    m = report.market
    base = plan.base_price
    stop_basis = "直近スイング安値" if plan.stop_basis == "swing_low" else "終値 − ATR"
    entry = "翌営業日の寄り付き"
    if plan.skip_if_open_at_or_above is not None:
        gap = format_price(plan.skip_if_open_at_or_above, m)
        entry += f"（{gap} 以上の窓開けで寄り付いたら見送り）"
    lines = [
        f"{i}. {j.symbol} {j.name} — {j.pattern.name}（v{j.pattern.version}）",
        f"  {_score(j, report)}",
        "  根拠:",
        *(_condition_line(c) for c in j.evaluation.conditions),
        f"  終値: {format_price(base, m)}",
        f"  エントリーの目安: {entry}",
        f"  損切りの目安: {format_price(plan.stop, m)}（{_pct(plan.stop, base)}、{stop_basis}）",
        f"  利確の目安: {format_price(plan.target, m)}（{_pct(plan.target, base)}）",
        f"  最大保有: {plan.max_holding_days}営業日",
    ]
    lines += [f"  注意: {note}" for note in j.outcome.notes]
    return lines


def _missed_keys(j: Judgement) -> str:
    keys = [KEYS[c.key].display_name for c in j.evaluation.conditions if c.status != "passed"]
    return "、".join(keys)


def email_body(report: Report) -> str:
    d = report.target_date
    market = MARKET_NAMES[report.market]
    lines = [f"{d:%Y-%m-%d}（{WEEKDAYS[d.weekday()]}）の{market}の判定結果です。"]
    lines += [
        "損切り・利確は当日終値を基準にした目安です。このツールは発注しません。"
        "チャートで形を確かめてから判断してください。",
        "",
    ]
    if report.signals:
        lines.append(f"■ シグナル {len(report.signals)}件（スコア順）")
        for i, j in enumerate(report.signals, 1):
            lines += ["", *_signal_block(i, j, report)]
    else:
        lines.append("■ シグナル: 該当なし")
    if report.near_misses:
        lines += ["", "■ 惜しかった銘柄（しきい値未満）"]
        for j in report.near_misses:
            lines.append(
                f"- {j.symbol} {j.name} — {j.pattern.name} {_score(j, report)}"
                f" — 未達: {_missed_keys(j)}"
            )
    lines += ["", f"■ 警告 {len(report.warnings)}件" if report.warnings else "■ 警告: なし"]
    lines += [f"- {w}" for w in report.warnings]
    return "\n".join(lines) + "\n"


def line_text(report: Report) -> str:
    m = report.market
    lines = [subject(report)]
    for j in report.signals:
        plan = j.evaluation.exit
        assert plan is not None
        lines += [
            "",
            f"{j.symbol} {j.name} {format_value(j.evaluation.score)}点",
            f"損切り {format_price(plan.stop, m)}（{_pct(plan.stop, plan.base_price)}）"
            f" / 利確 {format_price(plan.target, m)}（{_pct(plan.target, plan.base_price)}）"
            f" / {plan.max_holding_days}日",
        ]
        lines += [f"注意: {note}" for note in j.outcome.notes]
    if report.near_misses:
        misses = "、".join(
            f"{j.symbol}（{format_value(j.evaluation.score)}点）" for j in report.near_misses
        )
        lines += ["", f"惜しい: {misses}"]
    if report.signals or report.warnings:
        lines += ["", "根拠と警告の詳細はメールで確認してください。"]
    text = "\n".join(lines)
    if len(text) > LINE_MAX_CHARS:
        text = text[: LINE_MAX_CHARS - 20] + "\n…（続きはメールで）"
    return text
