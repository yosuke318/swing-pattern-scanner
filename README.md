# swing-pattern-scanner

日足OHLCVから短期スイング向けの「良い形」の銘柄をルールで検知し、出口の目安付きでメール／LINEに通知する個人用バッチ。GitHub Actions で動かし、DB は使わない。

要件と設計は Linear のプロジェクト「投資判断支援ツール」にある要件定義書 v2.0・基本設計書 v2.0 を参照。

## ディレクトリ構成

```text
config/
  settings.yaml         取得日数・流動性下限・スコアしきい値・クールダウン・通知・市場環境フィルタ
  symbols.yaml          監視銘柄（外すときは watched: false）
  events.yaml           決算日など（手動登録）
  patterns/<pattern_id>/v<n>.yaml   パターン定義（過去バージョンも残す）
state/                  日次実行で更新しコミットする
  notified.json         通知履歴 "<symbol>|<pattern_id>" → 最終通知日
  runs.csv              実行ログ
  results/YYYY-MM.csv   判定ログ（判定対象日の月別）
reports/backtest/       バックテスト結果
.cache/                 バックテスト用の過去データ（コミットしない）
src/swing_scanner/      コード
```

## 開発

[uv](https://docs.astral.sh/uv/) を使う。

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

設定ファイルを書き換えたら、スキーマ検証を通す。不正な設定（必須項目の欠落、型違い、未知のキー）はエラーになる。

```bash
uv run swing-scanner validate-config
```

## 日次実行

```bash
uv run swing-scanner run --market JP              # 対象日は現地時刻で直近に引けた日
uv run swing-scanner run --market US --date 2026-11-02   # 対象日を指定（手動実行）
```

- 実行の先頭で取引カレンダー（日本株は J-Quants、米国株は Alpaca）を引き、対象日が休場日なら `runs.csv` に `skipped_holiday` を記録して正常終了する。休場日リストは自前で持たない。
- 米国株は日本時間の翌朝に動くので、対象日は NY 時間の日付になる。日本の祝日でも米国株は通常どおり判定する。
- カレンダーを取得・解釈できなければ異常終了する。
- 認証情報は環境変数から読む。J-Quants は `JQUANTS_API_KEY`（v2）、または `JQUANTS_REFRESH_TOKEN` か `JQUANTS_MAIL` + `JQUANTS_PASSWORD`（v1）。Alpaca は `APCA_API_KEY_ID` + `APCA_API_SECRET_KEY`。

## 指標計算の決まり

`src/swing_scanner/features/` にある。日次処理とバックテストで同じ関数を使う。計算方式を変えるとバックテスト結果が再現できなくなるので、変えるときはここも直す。

- 移動平均は単純移動平均（ma5 / ma25 / ma75 / ma200）。
- ゴールデンクロスは「前日 MA短期 ≤ MA長期、当日 MA短期 > MA長期」。既定の組は 25 / 75。`days_since_golden_cross` は直近の GD からの営業日数（当日 0）で、まだ GD が無ければ `inf`。
- ATR14 は Wilder の平滑化（最初の14日の単純平均を初期値にする。TradingView の `ta.atr(14)` と同じ）。True Range は前日終値との窓を含む。
- ADX14 / +DI / -DI は TradingView の `ta.dmi(14, 14)` と同じ定義（Wilder の平滑化）。値動きが全く無い期間は 0。
- `atr14_pct` は ATR14 ÷ 終値の比率（0.023 = 2.3%）。`atr252_percentile` は当日を含む直近252営業日の `atr14_pct` の中での順位（0〜100）。
- 計算期間が足りない値は NaN（評価不能）。上場から252営業日未満の銘柄は `atr252_percentile` が NaN になる。

## 設定ファイルの決まり

- `symbols.yaml` の `symbol` は、日本株は4桁のコードを文字列で書く（`"7203"`）。米国株はティッカー（`BRK.B` など）。形式違いと、同じ市場・同じコードの重複はエラー。
- 監視銘柄の追加は行を足してコミットする（GitHub の Web 編集でもよい）。除外は行を消さず `watched: false` にする。
- 指数ETF（1306 / SPY）は `symbols.yaml` に入れず、`settings.yaml` の `markets.<市場>.benchmark_symbol` で持つ。通知対象にはしない。
- `events.yaml` には `symbols.yaml` にある銘柄だけを書ける。
- パターン定義のファイル名は `v<version>.yaml`、ディレクトリ名は `pattern_id` と一致させる。同じ `pattern_id` で `active: true` は1バージョンだけ。
- `conditions` の `weight` の合計は 100。`between` の `value` は `[下限, 上限]`、真偽値は `==` / `!=` でだけ比べる。
- `score_threshold` を省略したパターンは `settings.yaml` の `scoring.default_score_threshold` を使う。
- 寄り付きの窓開けでエントリーを見送るしきい値（g×ATR）は `exit.skip_gap_atr_mult` に書く。省略時は見送らない。
- 認証情報と通知先のアドレス・トークンは GitHub Actions Secrets に置き、リポジトリには書かない。

## 状態ファイルの書き込み

日本株と米国株のジョブはどちらも同じ `state/` を書く。ファイルを市場別に分けず、ワークフローの `concurrency` を共通のグループにして直列化する。
