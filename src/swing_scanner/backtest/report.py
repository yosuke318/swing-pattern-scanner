"""バックテストのレポート（[4-1]）。reports/backtest/<実行日時>_<pattern_id>_v<n>.md と、
トレード一覧の CSV を出す。

レポートの「実行条件」から同じ結果を再現できるよう、パターン定義・期間・銘柄・コスト・指標と
フィルタの設定・コードのコミット・データの取得日時をすべて書く。詳しい評価指標（最大ドローダウン、
レジーム別の内訳、バージョン比較）は [4-4] で足す。
"""

import csv
import datetime as dt
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from swing_scanner.backtest.runner import BacktestResult
from swing_scanner.backtest.trades import Trade
from swing_scanner.config import Config

DEFAULT_REPORT_DIR = Path("reports/backtest")


@dataclass(frozen=True)
class Summary:
    signals: int
    trades: int
    skipped: dict[str, int]
    wins: int
    win_rate: float | None
    avg_net_return: float | None
    median_net_return: float | None
    profit_factor: float | None
    avg_holding_days: float | None
    exit_reasons: dict[str, int]
    avg_mfe: float | None
    avg_mae: float | None


def summarize(result: BacktestResult) -> Summary:
    trades = result.trades
    returns = sorted(t.net_return for t in trades)
    gains = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    skipped: dict[str, int] = {}
    for s in result.signals:
        if s.skip_reason:
            skipped[s.skip_reason] = skipped.get(s.skip_reason, 0) + 1
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    n = len(trades)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return Summary(
        signals=len(result.signals),
        trades=n,
        skipped=skipped,
        wins=sum(1 for r in returns if r > 0),
        win_rate=(sum(1 for r in returns if r > 0) / n) if n else None,
        avg_net_return=mean(returns),
        median_net_return=returns[n // 2] if n else None,
        profit_factor=(gains / losses) if losses > 0 else None,
        avg_holding_days=mean([t.holding_days for t in trades]),
        exit_reasons=reasons,
        avg_mfe=mean([t.mfe for t in trades]),
        avg_mae=mean([t.mae for t in trades]),
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.2f}%"


def _rate(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


SKIP_LABELS = {
    "gap": "窓開けで見送り",
    "holding": "保有中で見送り",
    "no_next_bar": "翌営業日のデータなし",
    "no_exit_plan": "出口を算出できない",
}
EXIT_LABELS = {"stop": "損切り", "target": "利確", "time": "時間切れ", "end_of_data": "期間終了"}


def render_markdown(
    result: BacktestResult,
    config: Config,
    command: str,
    commit_sha: str,
    generated_at: dt.datetime,
) -> str:
    p = result.pattern
    s = summarize(result)
    settings = config.settings
    lines = [
        f"# バックテスト: {p.name}（{p.pattern_id} v{p.version}）",
        "",
        f"- 市場: {result.market}、期間: {result.start} 〜 {result.end}",
        f"- 生成: {generated_at.isoformat(timespec='seconds')}、コミット: `{commit_sha}`",
        f"- コマンド: `{command}`",
        "",
        "## 結果",
        "",
        "| 項目 | 値 |",
        "| -- | -- |",
        f"| 通知対象のシグナル | {s.signals} |",
        f"| トレード | {s.trades} |",
        f"| 勝ち（コスト控除後 > 0） | {s.wins}（勝率 {_rate(s.win_rate)}） |",
        f"| 平均損益（コスト控除後） | {_pct(s.avg_net_return)} |",
        f"| 損益の中央値 | {_pct(s.median_net_return)} |",
        f"| プロフィットファクター | {_num(s.profit_factor)} |",
        f"| 平均保有日数 | {_num(s.avg_holding_days, 1)} |",
        f"| 平均 MFE / MAE | {_pct(s.avg_mfe)} / {_pct(s.avg_mae)} |",
        "",
        "決済理由: "
        + (
            "、".join(f"{EXIT_LABELS.get(k, k)} {v}" for k, v in sorted(s.exit_reasons.items()))
            or "なし"
        ),
        "",
        "トレードにしなかったシグナル: "
        + (
            "、".join(f"{SKIP_LABELS.get(k, k)} {v}" for k, v in sorted(s.skipped.items()))
            or "なし"
        ),
        "",
        f"通知されなかった候補: クールダウン中 {result.cooled_down}"
        + "".join(f"、{k}で除外 {v}" for k, v in sorted(result.excluded.items())),
        "",
        "## 実行条件",
        "",
        f"- スコアのしきい値: {result.threshold:g}",
        f"- 想定コスト（片道）: {result.cost_per_side:.4%}",
        f"- クールダウン: {settings.notification.cooldown_business_days} 営業日",
        "- エントリー: シグナル翌営業日の寄り付き。損切り・利確はエントリー価格を基準に計算し直す。"
        "同じ日に損切りと利確の両方に届いたら損切りが先",
        "",
        "パターン定義:",
        "",
        "```yaml",
        yaml.safe_dump(p.model_dump(mode="json"), allow_unicode=True, sort_keys=False).rstrip(),
        "```",
        "",
        "指標・フィルタの設定（settings.yaml）:",
        "",
        "```yaml",
        yaml.safe_dump(
            {
                "fetch": settings.fetch.model_dump(mode="json"),
                "features": settings.features.model_dump(mode="json"),
                "filters": settings.filters.model_dump(mode="json"),
                "market_filter": settings.market_filter.model_dump(mode="json"),
                "markets": {result.market: settings.markets[result.market].model_dump(mode="json")},
            },
            allow_unicode=True,
            sort_keys=False,
        ).rstrip(),
        "```",
        "",
        "## データ",
        "",
        "| 銘柄 | 行数 | 取得日時 | キャッシュ | 注記 |",
        "| -- | -- | -- | -- | -- |",
    ]
    for d in [result.benchmark, *result.data]:
        fetched = d.fetched_at.isoformat(timespec="seconds") if d.fetched_at else "—"
        note = d.error or "；".join(d.warnings)
        label = f"{d.symbol}（ベンチマーク）" if d is result.benchmark else d.symbol
        cached = "使用" if d.from_cache else "取得"
        lines.append(f"| {label} | {d.rows} | {fetched} | {cached} | {note} |")
    lines += [
        "",
        "## 注意",
        "",
        "- 対象銘柄は今の監視銘柄から選んでいるので、生存バイアスがある（今も監視している＝"
        "うまくいった銘柄に偏る）。数字は実態より良く出る方向にずれうる",
        "- 価格は取得時点の調整済み価格。取得後に株式分割があるとキャッシュと食い違うので、"
        "ときどき `--refresh-cache` で取り直す",
        "",
    ]
    return "\n".join(lines)


TRADE_COLUMNS = [f.name for f in fields(Trade)]


def write_report(
    result: BacktestResult,
    config: Config,
    command: str,
    commit_sha: str,
    generated_at: dt.datetime,
    out_dir: Path = DEFAULT_REPORT_DIR,
) -> tuple[Path, Path]:
    stem = f"{generated_at:%Y%m%d-%H%M%S}_{result.pattern.pattern_id}_v{result.pattern.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{stem}.md"
    md.write_text(
        render_markdown(result, config, command, commit_sha, generated_at), encoding="utf-8"
    )
    trades_csv = out_dir / f"{stem}_trades.csv"
    with trades_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for t in result.trades:
            writer.writerow(asdict(t))
    return md, trades_csv
