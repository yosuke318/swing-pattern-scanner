"""取得データの検証（要件 5.6、[1-6]）。気づけない壊れ方を潰す。

検証は取得のたびに銘柄ごとに行い、結果を ValidationResult で返す。

判定から外すもの（status が ok 以外）:
  error    API エラー（リトライしても復旧しなかった。リトライは sources/http.py）
  missing  データが無い、必須の列が無い、対象日の行が無い（データ未更新）、対象日に売買が無い
  anomaly  論理矛盾（高値 < 安値、高値 < 始値・終値、安値 > 始値・終値、価格 <= 0、出来高 < 0）、
           日付の重複・逆順
警告に留めるもの（status は ok のまま、warnings に積む）:
  - 過去の売買なしの日（J-Quants は四本値が null で収録する）。その行は除いて計算する
    （TradingView と同じく、売買の無い日には足を作らない）
  - 期間中の歯抜け（取引カレンダーの立会日なのに行が無い）
  - 過去の出来高 0 の日
  - 前日比 ±JUMP_THRESHOLD を超える終値の変化（ストップ高・安の日は除く）。調整漏れの検知も兼ねる。
    ストップ高・安を誤って異常にしないよう、判定からは外さない
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from swing_scanner.sources.base import OHLCV_COLUMNS

Status = Literal["ok", "error", "missing", "anomaly"]

# 前日比の終値の変化がこれを超えたら警告する（日本株の値幅制限は低位株で ±30% 前後まである）
JUMP_THRESHOLD = 0.35
PRICE_COLUMNS = ["open", "high", "low", "close"]
# 警告に並べる日付の数の上限
MAX_DATES_IN_MESSAGE = 3


@dataclass(frozen=True)
class ValidationResult:
    status: Status
    # 判定に使うデータ（売買なしの日を除いたもの）。ok でなければ None
    data: pd.DataFrame | None
    # ok でないときの理由
    reason: str = ""
    # 判定には使うが、知らせておくこと
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def failed(status: Status, reason: str) -> ValidationResult:
    return ValidationResult(status, None, reason)


def _dates(dates: Sequence[dt.date]) -> str:
    shown = ", ".join(str(d) for d in list(dates)[:MAX_DATES_IN_MESSAGE])
    return shown + (" など" if len(dates) > MAX_DATES_IN_MESSAGE else "")


def validate_ohlcv(
    df: pd.DataFrame, target_date: dt.date, sessions: Sequence[dt.date] | None = None
) -> ValidationResult:
    """1銘柄の日足を検証する。sessions（取引カレンダーの立会日）があれば歯抜けも見る。"""
    missing_columns = [c for c in OHLCV_COLUMNS if c not in df]
    if missing_columns:
        return failed("missing", f"列がありません: {missing_columns}")
    if df.empty:
        return failed("missing", "データがありません")
    dates = pd.Series(df["date"])
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        return failed("anomaly", "日付の並びが不正です（重複か逆順）")
    last = dates.iloc[-1]
    if last != target_date:
        return failed("missing", f"{target_date} のデータがありません（最終 {last}。データ未更新）")

    no_trade = df[PRICE_COLUMNS].isna().all(axis=1)
    if no_trade.iloc[-1] or not (df["volume"].iloc[-1] > 0):
        return failed("missing", f"{target_date} に売買がありません")
    traded = df[~no_trade].reset_index(drop=True)
    if traded[OHLCV_COLUMNS[1:]].isna().any().any():
        return failed("missing", "四本値・出来高の一部が欠けています")

    o, h, lo, c = (traded[col] for col in PRICE_COLUMNS)
    broken = (
        (traded[PRICE_COLUMNS] <= 0).any(axis=1)
        | (h < lo)
        | (h < o)
        | (h < c)
        | (lo > o)
        | (lo > c)
        | (traded["volume"] < 0)
    )
    if broken.any():
        bad = traded.loc[broken, "date"].tolist()
        return failed("anomaly", f"四本値・出来高が矛盾しています（{_dates(bad)}）")

    warnings: list[str] = []
    if no_trade.any():
        days = df.loc[no_trade, "date"].tolist()
        warnings.append(f"売買なしの日が{len(days)}日あります（{_dates(days)}。除いて計算）")
    if sessions is not None:
        have = set(df["date"])
        gaps = [d for d in sessions if dates.iloc[0] <= d <= target_date and d not in have]
        if gaps:
            warnings.append(f"立会日の行が{len(gaps)}日ありません（{_dates(gaps)}）")
    zero_volume = traded.loc[traded["volume"] == 0, "date"].tolist()
    if zero_volume:
        warnings.append(f"出来高 0 の日が{len(zero_volume)}日あります（{_dates(zero_volume)}）")
    change = (c / c.shift(1) - 1).abs()
    jump = change > JUMP_THRESHOLD
    for flag in ("upper_limit", "lower_limit"):
        if flag in traded:
            jump &= ~traded[flag].fillna(False).astype(bool)
    if jump.any():
        days = traded.loc[jump, "date"].tolist()
        warnings.append(
            f"前日比 ±{JUMP_THRESHOLD:.0%} を超える値動きが{len(days)}日あります"
            f"（{_dates(days)}。調整漏れの可能性）"
        )
    return ValidationResult("ok", traded, warnings=warnings)
