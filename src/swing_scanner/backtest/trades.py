"""仮想トレードと決済（[4-2]、基本設計書 4.6）。通知で示す出口ルールと同じルールで決済する。

  エントリー: シグナル翌営業日の寄り付き（シグナル日の終値では入らない。終値は引け後に確定するので、
              終値で入ると実運用では不可能な取引になる）
    - 寄り付きが シグナル日の終値 + g×ATR 以上の窓開けなら見送る
      （skip_gap_atr_mult が無ければ見送らない）
  損切り:     直近スイング安値 と エントリー − k×ATR のうち浅い方
              （実際のエントリー価格を基準に計算し直す）
  利確:       エントリー + m×ATR
  時間切れ:   エントリー日から max_holding_days 営業日たったら、翌営業日の寄り付きで決済

- 日足の高値・安値で、損切り・利確に届いたかを毎日判定する
- 同じ日に損切りと利確の両方に届いたら、損切りが先とみなす（日足では順序が分からないので保守的に）
- 2日目以降、損切り価格以下で寄り付いたら（ギャップダウン）寄り付きで決済する。利確価格以上で
  寄り付いたら寄り付きで利確する
- データの最終日までに決済しなければ、最終日の終値で end_of_data として閉じる
- 損益は片道コスト c を両側にかけて求める: 決済 × (1 − c) ÷ (エントリー × (1 + c)) − 1
- MFE / MAE は保有中の最大含み益・最大含み損（エントリー価格比）。日中に決済した日は決済価格だけを
  使う（その日の高値・安値と決済の前後が分からないため）

ATR とスイング安値はシグナル日の値（その日までに確定したもの）を使う。エントリー価格と決済は
シグナル日より後の足だけで決まる。
"""

import datetime as dt
import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from swing_scanner.config import ExitRule
from swing_scanner.rules.engine import exit_plan

ExitReason = Literal["stop", "target", "time", "end_of_data"]
SkipReason = Literal["gap", "no_next_bar", "no_exit_plan"]


@dataclass(frozen=True)
class Trade:
    symbol: str
    pattern_id: str
    version: int
    signal_date: dt.date
    entry_date: dt.date
    entry_price: float
    stop: float
    stop_basis: str
    target: float
    exit_date: dt.date
    exit_price: float
    exit_reason: ExitReason
    # エントリー日から決済日までの営業日数（同じ日に決済したら 0）
    holding_days: int
    gross_return: float
    net_return: float
    mfe: float
    mae: float


def simulate_trade(
    bars: pd.DataFrame,
    signal_idx: int,
    rule: ExitRule,
    atr14: float,
    swing_low: float | None,
    cost_per_side: float,
    symbol: str,
    pattern_id: str,
    version: int,
) -> Trade | SkipReason:
    """signal_idx 行目のシグナルを決済する。bars は date / open / high / low / close の昇順。"""
    entry_idx = signal_idx + 1
    if entry_idx >= len(bars):
        return "no_next_bar"
    o, h, lo, c = (bars[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close"))
    dates = bars["date"].tolist()
    if atr14 is None or math.isnan(atr14):
        return "no_exit_plan"
    gap = rule.skip_gap_atr_mult
    if gap is not None and o[entry_idx] >= c[signal_idx] + gap * atr14:
        return "gap"
    entry = o[entry_idx]
    plan = exit_plan(rule, entry, atr14, swing_low)
    if plan is None:
        return "no_exit_plan"
    stop, target = plan.stop, plan.target

    best, worst = entry, entry
    exit_idx, exit_price, reason = len(bars) - 1, c[-1], "end_of_data"
    for j in range(entry_idx, len(bars)):
        if j > entry_idx and o[j] <= stop:
            exit_idx, exit_price, reason = j, o[j], "stop"
            break
        if j > entry_idx and o[j] >= target:
            exit_idx, exit_price, reason = j, o[j], "target"
            break
        if lo[j] <= stop:
            exit_idx, exit_price, reason = j, stop, "stop"
            break
        if h[j] >= target:
            exit_idx, exit_price, reason = j, target, "target"
            break
        best, worst = max(best, h[j]), min(worst, lo[j])
        if j - entry_idx + 1 >= rule.max_holding_days:
            if j + 1 < len(bars):
                exit_idx, exit_price, reason = j + 1, o[j + 1], "time"
            else:
                exit_idx, exit_price, reason = j, c[j], "end_of_data"
            break
    best, worst = max(best, exit_price), min(worst, exit_price)

    gross = exit_price / entry - 1
    net = exit_price * (1 - cost_per_side) / (entry * (1 + cost_per_side)) - 1
    return Trade(
        symbol=symbol,
        pattern_id=pattern_id,
        version=version,
        signal_date=dates[signal_idx],
        entry_date=dates[entry_idx],
        entry_price=entry,
        stop=stop,
        stop_basis=plan.stop_basis,
        target=target,
        exit_date=dates[exit_idx],
        exit_price=float(exit_price),
        exit_reason=reason,  # type: ignore[arg-type]
        holding_days=exit_idx - entry_idx,
        gross_return=gross,
        net_return=net,
        mfe=best / entry - 1,
        mae=worst / entry - 1,
    )
