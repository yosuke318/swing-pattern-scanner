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

`src/swing_scanner/features/` にある。入口は `compute_features(ohlcv, benchmark, settings.features)` で、日次処理（約300営業日を入れて最終行を使う）とバックテスト（最大5年を入れて全行を使う）が同じ関数を使う。出力の列は基本設計書 3.6 の列のあとに、判定やログで使う補助の列（`dead_cross_flag`、`prior_swing_high_price`、`swing_low_higher_than_prior`、`close_minus_ma<n>_atr_ratio`）が並ぶ。

パラメータ（ゴールデンクロスの組、スイングの N、HH/HL で比べる個数、`trend_regime` の方式、ヒステリシス）は `settings.yaml` の `features` に書く。計算方式を変えるとバックテスト結果が再現できなくなるので、変えるときはここも直す。

- 移動平均は単純移動平均（ma5 / ma25 / ma75 / ma200）。
- ゴールデンクロスは「前日 MA短期 ≤ MA長期、当日 MA短期 > MA長期」。組は `features.golden_cross_pair`（25 / 75）。`days_since_golden_cross` は直近の GD からの営業日数（当日 0）で、まだ GD が無ければ `inf`。
- ATR14 は Wilder の平滑化（最初の14日の単純平均を初期値にする。TradingView の `ta.atr(14)` と同じ）。True Range は前日終値との窓を含む。
- ADX14 / +DI / -DI は TradingView の `ta.dmi(14, 14)` と同じ定義（Wilder の平滑化）。値動きが全く無い期間は 0。
- `atr14_pct` は ATR14 ÷ 終値の比率（0.023 = 2.3%）。`atr252_percentile` は当日を含む直近252営業日の `atr14_pct` の中での順位（0〜100）。
- スイング高値は「高値が前 N 日より高く、後 N 日以上」の日（安値も同様）。N は固定値で `features.swing_window`（3）。バックテストで比べて決める。i 日目のスイングは i+N 日目に確定したものとして扱い、各行はその日までに確定したスイングだけを使う。
- `hh_hl_structure` は直近2つのスイング高値・安値がともに切り上がれば uptrend、ともに切り下がれば downtrend、それ以外は unclear。
- `vol_regime` は `atr252_percentile` が 20 以下で low、80 以上で high、それ以外は normal。
- `trend_regime` は ADX（ADX14 ≥ 20 と ±DI の向き）・HH/HL 構造・MA の並び（終値・MA25・MA75）の3つのうち、2つ以上が同じ向きで逆向きが無ければその向き、それ以外は range（`features.trend_method: vote`）。単独の方式（adx / structure / ma）にも切り替えられ、実データの該当日数を比べて決め直す。区分が連続 k 日続くまで切り替えないヒステリシス（`features.hysteresis_days`）は既定で無し（1）。
- `trading_value_20d` は 20 日平均売買代金。API の売買代金（`trading_value` 列）を使い、無ければ終値 × 出来高。
- `volume_ratio_20d` は当日出来高 ÷ 前日までの20日平均出来高（当日を分母に入れない）。
- `relative_strength_20d` は銘柄と指数ETFの20日リターンの差。指数を取得できなかった日や、指数に無い日付は NaN（評価不能）。
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

## パターン定義と判定

パターン定義は `config/patterns/<pattern_id>/v<n>.yaml` に置く（書式は基本設計書 3.3、上の「設定ファイルの決まり」も参照）。

- 本番に適用するバージョンは `active: true` の1つだけ。切り替えは active を書き換えるコミットで行い、戻すときは revert する。
- 一度使ったバージョンのファイルは書き換えない。閾値を変えるときは新しいバージョンのファイルを作る（過去の判定ログがどの閾値で出たか分からなくなるため）。
- バックテストでは `Config.pattern(pattern_id, version)` で過去のバージョンを指定して読み込む。
- `key` は下の一覧にあるものだけ使える。一覧に無い key、真偽値の key に数値を書く（またはその逆）と読み込み時にエラーになる。key は一度使ったら消さず、意味も変えない。

### パターン条件で使える key

`src/swing_scanner/rules/keys.py` で定義している。一覧を変えたらこの表も直す（テストで突き合わせる）。

| key | 型 | 内容 |
| -- | -- | -- |
| `close` | 数値 | 終値 |
| `ma5` | 数値 | 5日単純移動平均 |
| `ma25` | 数値 | 25日単純移動平均 |
| `ma75` | 数値 | 75日単純移動平均 |
| `ma200` | 数値 | 200日単純移動平均 |
| `close_minus_ma5_atr_ratio` | 数値 | (終値 − MA5) ÷ ATR14 |
| `close_minus_ma25_atr_ratio` | 数値 | (終値 − MA25) ÷ ATR14 |
| `close_minus_ma75_atr_ratio` | 数値 | (終値 − MA75) ÷ ATR14 |
| `close_minus_ma200_atr_ratio` | 数値 | (終値 − MA200) ÷ ATR14 |
| `golden_cross_within_days` | 数値 | 直近のゴールデンクロスからの営業日数（当日 0、未発生は inf）。`<=` で「X日以内」 |
| `golden_cross_flag` | 真偽値 | 当日ゴールデンクロスが発生した |
| `adx14` | 数値 | ADX14（トレンドの強さ） |
| `plus_di` | 数値 | +DI |
| `minus_di` | 数値 | -DI |
| `plus_di_above_minus_di` | 真偽値 | +DI > -DI（上向きのトレンド） |
| `swing_low_higher_than_prior` | 真偽値 | 直近の確定スイング安値 > その前の安値 |
| `swing_high_higher_than_prior` | 真偽値 | 直近の確定スイング高値 > その前の高値 |
| `hh_hl_uptrend` | 真偽値 | hh_hl_structure が uptrend |
| `atr14` | 数値 | ATR14（Wilder） |
| `atr14_pct` | 数値 | ATR14 ÷ 終値（比率） |
| `atr252_percentile` | 数値 | atr14_pct の直近252日での順位（0〜100） |
| `volume_ratio_20d` | 数値 | 当日出来高 ÷ 前日までの20日平均出来高 |
| `relative_strength_20d` | 数値 | 20日リターンの指数ETFとの差（比率） |
| `trading_value_20d` | 数値 | 20日平均売買代金 |

### 判定（ルールエンジン）

`src/swing_scanner/rules/engine.py` の `evaluate(pattern, row)`。日次処理とバックテストで同じ関数を使う。

- 条件ごとに passed / failed / unevaluable（評価不能）を返す。指標が欠けている条件は評価不能で、スコアに入らず、全条件合致（matched）にもならない。
- `between` は両端を含む（下限 ≤ 値 ≤ 上限）。
- スコアは passed の条件の weight の合計（100点満点）。matched は全条件 passed で、かつレジームが対象のとき。
- パターンの `regimes` に当日の `vol_regime` / `trend_regime` が無ければ対象外（matched にせず、理由を残す）。レジームが評価不能のときも対象外。
- 出口（基本設計書 4.6）は当日終値を基準にした目安。損切りは「直近スイング安値」と「終値 − k×ATR」の浅い方（スイング安値が無い・終値以上なら ATR 側）、利確は「終値 + m×ATR」。バックテストは `exit_plan` にエントリー価格を渡して計算し直す。

## 状態ファイルの書き込み

日本株と米国株のジョブはどちらも同じ `state/` を書く。ファイルを市場別に分けず、ワークフローの `concurrency` を共通のグループにして直列化する。
