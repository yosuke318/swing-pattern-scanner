"""出来高比率と市場相対強度（基本設計書 4.5）。

volume_ratio_20d = 当日出来高 ÷ 前日までの20日平均出来高。当日を分母に含めると、出来高が急増した
  日ほど分母も膨らんで比率が鈍るので、前日までで平均する。平均が 0（20日間売買が無い）なら NaN。
relative_strength_20d = 銘柄の20日リターン − 指数ETFの20日リターン（比率の差。0.03 なら 3%pt）。
  指数は銘柄の日付に合わせて並べ、両方とも同じ20行前の終値と比べる。指数に無い日付や、指数を
  取得できなかった日（benchmark が None）は NaN（評価不能）にする。

米国株の出来高は Alpaca 無料プランの IEX フィードだけなので実際の一部しか映らない。比率なら偏りが
分母分子で打ち消し合う可能性はあるが、日本株と同じ閾値を当てるかは検証してから決める。
"""

import pandas as pd

VOLUME_WINDOW = 20
RETURN_DAYS = 20


def volume_ratio(volume: pd.Series, window: int = VOLUME_WINDOW) -> pd.Series:
    average = volume.astype(float).shift(1).rolling(window, min_periods=window).mean()
    return volume / average.where(average > 0)


def relative_strength(
    dates: pd.Series,
    close: pd.Series,
    benchmark: pd.DataFrame | None,
    days: int = RETURN_DAYS,
) -> pd.Series:
    if benchmark is None:
        return pd.Series(float("nan"), index=close.index)
    bench_close = dates.map(benchmark.set_index("date")["close"]).astype(float)
    stock_return = close / close.shift(days) - 1
    bench_return = bench_close / bench_close.shift(days) - 1
    return stock_return - bench_return


def add_relative(df: pd.DataFrame, benchmark: pd.DataFrame | None) -> pd.DataFrame:
    """volume_ratio_20d / relative_strength_20d を足す。"""
    out = df.copy()
    out["volume_ratio_20d"] = volume_ratio(out["volume"])
    out["relative_strength_20d"] = relative_strength(out["date"], out["close"], benchmark)
    return out
