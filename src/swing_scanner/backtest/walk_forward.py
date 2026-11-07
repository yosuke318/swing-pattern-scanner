"""ウォークフォワード検証（[4-3]、要件 5.7）。

閾値と出口を全期間で調整してから同じ期間で成績を測ると、必ず良い数字が出る（過学習）。
学習期間で閾値と出口を決め、その後ろの評価期間（未知の期間）で成績を測る。これを時系列を
ずらしながら繰り返し、学習期間と評価期間の成績を並べる（乖離の大きさが過学習の度合い）。

  学習[2021-01〜2022-12] → 評価[2023-01〜2023-12]
  学習[2022-01〜2023-12] → 評価[2024-01〜2024-12]  …

- 探索するのは スコアのしきい値・損切りの ATR 倍率・利確の ATR 倍率・最大保有日数 だけ。
  しきい値の候補はパターンの weight の組み合わせで取りうる 50 点以上の点数、出口の候補は
  settings.yaml の backtest.walk_forward
- 学習期間で選ぶのは、トレード数が min_trades 以上の組み合わせのうち、1トレードあたりの
  平均損益（コスト控除後）が最も高いもの。同点ならトレード数が多い方。どれも min_trades に
  届かなければ、その窓は「選べない」として評価しない
- 学習期間のトレードは学習期間の終わりより後の足を使わない（終わりまでに決済しなければ終わりの
  終値で閉じる）。評価期間の値動きが探索に漏れないようにするため
- 評価期間のトレード数が min_trades 未満なら「サンプル不足」と表示する
"""

import calendar
import csv
import datetime as dt
import itertools
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from swing_scanner.backtest.cache import OhlcvCache
from swing_scanner.backtest.report import TRADE_COLUMNS, Summary, summarize, summarize_trades
from swing_scanner.backtest.runner import BacktestResult, Prepared, prepare, simulate
from swing_scanner.backtest.trades import Trade
from swing_scanner.config import Config, Market, Pattern, Symbol

MIN_THRESHOLD = 50


def add_months(day: dt.date, months: int) -> dt.date:
    """months か月後の同じ日（その月に無ければ末日）。"""
    year, month = divmod(day.year * 12 + day.month - 1 + months, 12)
    month += 1
    return dt.date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


@dataclass(frozen=True)
class Window:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date


def make_windows(
    start: dt.date, end: dt.date, train_months: int, test_months: int, step_months: int
) -> list[Window]:
    """start から学習 train_months か月・評価 test_months か月の窓を step_months か月ずつずらす。

    評価期間が end を越える窓は end で切る。評価期間が始まらない窓は作らない。
    """
    if min(train_months, test_months, step_months) < 1:
        raise ValueError("学習期間・評価期間・ずらす幅は 1 か月以上にしてください")
    windows = []
    k = 0
    while True:
        train_start = add_months(start, k * step_months)
        test_start = add_months(train_start, train_months)
        if test_start > end:
            return windows
        test_end = min(add_months(test_start, test_months) - dt.timedelta(days=1), end)
        windows.append(Window(train_start, test_start - dt.timedelta(days=1), test_start, test_end))
        k += 1


@dataclass(frozen=True)
class Params:
    score_threshold: float
    stop_atr_mult: float
    target_atr_mult: float
    max_holding_days: int

    def apply(self, pattern: Pattern) -> Pattern:
        exit_rule = pattern.exit.model_copy(
            update={
                "stop": pattern.exit.stop.model_copy(update={"atr_mult": self.stop_atr_mult}),
                "target": pattern.exit.target.model_copy(update={"atr_mult": self.target_atr_mult}),
                "max_holding_days": self.max_holding_days,
            }
        )
        return pattern.model_copy(
            update={"score_threshold": self.score_threshold, "exit": exit_rule}
        )

    def label(self) -> str:
        return (
            f"しきい値 {self.score_threshold:g} / 損切り {self.stop_atr_mult:g}ATR / "
            f"利確 {self.target_atr_mult:g}ATR / 最大 {self.max_holding_days}日"
        )


def achievable_scores(pattern: Pattern) -> list[float]:
    """weight の組み合わせで取りうる MIN_THRESHOLD 点以上の点数。"""
    weights = [float(c.weight) for c in pattern.conditions]
    sums = {
        round(sum(combo), 9)
        for r in range(1, len(weights) + 1)
        for combo in itertools.combinations(weights, r)
    }
    return sorted(s for s in sums if s >= MIN_THRESHOLD)


def candidate_params(config: Config, pattern: Pattern) -> list[Params]:
    grid = config.settings.backtest.walk_forward
    return [
        Params(t, s, m, d)
        for t in achievable_scores(pattern)
        for s in grid.stop_atr_mult
        for m in grid.target_atr_mult
        for d in grid.max_holding_days
    ]


def _mean_return(trades: list[Trade]) -> float:
    return sum(t.net_return for t in trades) / len(trades)


@dataclass(frozen=True)
class WindowResult:
    window: Window
    # 学習期間で選んだもの。選べなかったら None
    params: Params | None
    train: BacktestResult | None
    test: BacktestResult | None
    tried: int

    @property
    def train_summary(self) -> Summary | None:
        return summarize(self.train) if self.train else None

    @property
    def test_summary(self) -> Summary | None:
        return summarize(self.test) if self.test else None


@dataclass(frozen=True)
class WalkForwardResult:
    pattern: Pattern
    market: Market
    start: dt.date
    end: dt.date
    train_months: int
    test_months: int
    step_months: int
    min_trades: int
    prepared: Prepared
    windows: list[WindowResult]

    @property
    def test_trades(self) -> list[Trade]:
        """評価期間のトレードをつないだもの（未知の期間での成績）。"""
        return [t for w in self.windows if w.test for t in w.test.trades]


def choose_params(
    config: Config, prepared: Prepared, pattern: Pattern, window: Window, min_trades: int
) -> tuple[Params | None, BacktestResult | None, int]:
    """学習期間だけで（学習期間の終わりより後の足を使わずに）閾値と出口を選ぶ。"""
    best: tuple[float, int] | None = None
    chosen: tuple[Params, BacktestResult] | None = None
    candidates = candidate_params(config, pattern)
    for params in candidates:
        result = simulate(
            config,
            prepared,
            params.apply(pattern),
            window.train_start,
            window.train_end,
            data_end=window.train_end,
        )
        if len(result.trades) < min_trades:
            continue
        key = (_mean_return(result.trades), len(result.trades))
        if best is None or key > best:
            best, chosen = key, (params, result)
    if chosen is None:
        return None, None, len(candidates)
    return chosen[0], chosen[1], len(candidates)


def run_walk_forward(
    config: Config,
    pattern: Pattern,
    market: Market,
    symbols: Sequence[Symbol],
    start: dt.date,
    end: dt.date,
    cache: OhlcvCache,
    train_months: int = 24,
    test_months: int = 12,
    step_months: int = 12,
    refresh: bool = False,
) -> WalkForwardResult:
    min_trades = config.settings.backtest.walk_forward.min_trades
    windows = make_windows(start, end, train_months, test_months, step_months)
    if not windows:
        raise ValueError("期間が短く、学習期間の後ろに評価期間を取れません")
    prepared = prepare(config, pattern, market, symbols, start, end, cache, refresh)
    results = []
    for w in windows:
        params, train, tried = choose_params(config, prepared, pattern, w, min_trades)
        test = None
        if params is not None:
            test = simulate(
                config, prepared, params.apply(pattern), w.test_start, w.test_end, w.test_end
            )
        results.append(WindowResult(w, params, train, test, tried))
    return WalkForwardResult(
        pattern,
        market,
        start,
        end,
        train_months,
        test_months,
        step_months,
        min_trades,
        prepared,
        results,
    )


# --- レポート ----------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _pf(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _cells(s: Summary | None, min_trades: int) -> str:
    if s is None:
        return "— | — | — | —"
    n = f"{s.trades}" + ("（サンプル不足）" if s.trades < min_trades else "")
    return f"{n} | {_pct(s.avg_net_return)} | {_rate(s.win_rate)} | {_pf(s.profit_factor)}"


def render_walk_forward(
    result: WalkForwardResult,
    command: str,
    commit_sha: str,
    generated_at: dt.datetime,
) -> str:
    p = result.pattern
    lines = [
        f"# ウォークフォワード検証: {p.name}（{p.pattern_id} v{p.version}）",
        "",
        f"- 市場: {result.market}、期間: {result.start} 〜 {result.end}",
        f"- 窓: 学習 {result.train_months} か月 → 評価 {result.test_months} か月、"
        f"{result.step_months} か月ずつずらす",
        f"- 信用する最低トレード数: {result.min_trades}（未満は「サンプル不足」）",
        f"- 生成: {generated_at.isoformat(timespec='seconds')}、コミット: `{commit_sha}`",
        f"- コマンド: `{command}`",
        "",
        "学習期間で、トレード数が最低件数以上の組み合わせのうち 1トレードあたりの平均損益"
        "（コスト控除後）が最も高いしきい値と出口を選び、その後ろの評価期間に当てた。"
        "学習期間のトレードは学習期間の終わりより後の値動きを使っていない。"
        "学習と評価の成績の乖離が大きいほど過学習している。",
        "",
        "## 窓ごとの結果",
        "",
        "| 学習期間 | 評価期間 | 選んだしきい値と出口 | 学習: 件数 | 平均 | 勝率 | PF "
        "| 評価: 件数 | 平均 | 勝率 | PF |",
        "| -- | -- | -- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for w in result.windows:
        win = w.window
        params = w.params.label() if w.params else f"選べない（{w.tried}通りとも件数不足）"
        lines.append(
            f"| {win.train_start}〜{win.train_end} | {win.test_start}〜{win.test_end} "
            f"| {params} | {_cells(w.train_summary, result.min_trades)} "
            f"| {_cells(w.test_summary, result.min_trades)} |"
        )
    oos = result.test_trades
    oos_summary = summarize_trades(oos) if oos else None
    lines += [
        "",
        "## 評価期間をつないだ成績（未知の期間での成績）",
        "",
        "| 件数 | 平均 | 勝率 | PF |",
        "| --: | --: | --: | --: |",
        f"| {_cells(oos_summary, result.min_trades)} |",
        "",
        "## 注意",
        "",
        "- 対象銘柄は今の監視銘柄なので生存バイアスがある",
        "- サンプル不足の窓の数字は信用しない。窓を広げるか、銘柄を増やして件数を確保する",
        "",
    ]
    return "\n".join(lines)


def write_walk_forward(
    result: WalkForwardResult,
    command: str,
    commit_sha: str,
    generated_at: dt.datetime,
    out_dir: Path,
) -> tuple[Path, Path]:
    p = result.pattern
    stem = f"{generated_at:%Y%m%d-%H%M%S}_{p.pattern_id}_v{p.version}_walkforward"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{stem}.md"
    md.write_text(render_walk_forward(result, command, commit_sha, generated_at), encoding="utf-8")
    trades_csv = out_dir / f"{stem}_test_trades.csv"
    with trades_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", *TRADE_COLUMNS], lineterminator="\n")
        writer.writeheader()
        for i, w in enumerate(result.windows, 1):
            for t in w.test.trades if w.test else []:
                writer.writerow({"window": i, **asdict(t)})
    return md, trades_csv
