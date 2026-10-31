"""日次処理のつなぎ方と失敗時の振る舞い（[1-10]）。"""

import csv
import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_features import ohlcv

from swing_scanner.cli import main
from swing_scanner.config import load_config
from swing_scanner.daily import DailyError, run_daily
from swing_scanner.notify.send import NotifyError
from swing_scanner.sources.base import SourceUnavailable
from swing_scanner.sources.http import ApiError
from swing_scanner.state import load_notified, notified_key
from swing_scanner.trading_calendar import WeekdayCalendar

ROOT = Path(__file__).resolve().parents[1]
DAY = dt.date(2026, 10, 30)  # 金曜
NOW = dt.datetime(2026, 10, 30, 10, 0, tzinfo=dt.UTC)  # JST 19:00
JP = ["7203", "6758", "8306", "8035", "9984"]

# レジームや条件にかかわらず、どの銘柄にも当たるパターン（通知の流れを確かめる用）
ALWAYS = {
    "pattern_id": "always",
    "version": 1,
    "name": "常に当たる",
    "category": "trend",
    "active": True,
    "regimes": {"vol": ["low", "normal", "high"], "trend": ["uptrend", "range", "downtrend"]},
    "conditions": [{"key": "close", "op": ">", "value": 0, "weight": 100}],
    "score_threshold": 100,
    "exit": {
        "stop": {"type": "atr", "atr_mult": 1.5},
        "target": {"atr_mult": 3},
        "max_holding_days": 5,
    },
}


def frame(seed: int, end: dt.date = DAY, n: int = 400) -> pd.DataFrame:
    df = ohlcv(n, seed=seed)
    df["date"] = pd.bdate_range(end=end, periods=n).date
    return df


class DictSource:
    def __init__(self, frames: dict[str, pd.DataFrame], fail: set[str] = frozenset()) -> None:
        self.frames, self.fail = frames, fail
        self.calls: list[tuple[str, dt.date, dt.date]] = []

    def fetch(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if symbol in self.fail:
            raise ApiError("HTTP 500")
        df = self.frames[symbol]
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


class FakeEmail:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def send(self, subject_text: str, body: str) -> None:
        if self.fail:
            raise NotifyError("SMTP down")
        self.sent.append((subject_text, body))


def source(**kw) -> DictSource:
    frames = {code: frame(i) for i, code in enumerate(JP)} | {"1306": frame(99)}
    return DictSource(frames, **kw)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    return tmp_path / "config"


def always_config(config_dir: Path):
    for path in (config_dir / "patterns").glob("*/*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["active"] = False
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (config_dir / "patterns/always").mkdir()
    (config_dir / "patterns/always/v1.yaml").write_text(yaml.safe_dump(ALWAYS), encoding="utf-8")
    return load_config(config_dir)


def run(config, state: Path, src=None, email=None, **kw):
    return run_daily(
        config,
        "JP",
        WeekdayCalendar("JP"),
        src or source(),
        state,
        now=kw.pop("now", NOW),
        commit_sha="abc123",
        email=email if email is not None else FakeEmail(),
        log=lambda m: None,
        **kw,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- 通しの実行 --------------------------------------------------------------------


def test_full_run_with_repo_patterns(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    email = FakeEmail()
    src = source()
    result = run(config, tmp_path, src, email)
    assert (result.target.date, result.status) == (DAY, "success")
    # 監視銘柄 × 有効なパターンの数だけ判定ログが残る
    rows = read_csv(tmp_path / "results/2026-10.csv")
    assert len(rows) == len(JP) * len(config.active_patterns())
    assert {r["commit_sha"] for r in rows} == {"abc123"}
    [run_row] = read_csv(tmp_path / "runs.csv")
    assert (run_row["status"], run_row["symbols_ok"], run_row["symbols_failed"]) == (
        "success",
        "5",
        "0",
    )
    # 該当なしの日もメールは1通送る
    assert len(email.sent) == 1 and email.sent[0][0].startswith("【JP】2026-10-30")
    # 取得は lookback_days 営業日を含む範囲で、ベンチマークも取る
    assert {c[0] for c in src.calls} == set(JP) | {"1306"}
    assert all(c[2] == DAY for c in src.calls)


def test_signals_are_sent_logged_and_recorded(config_dir: Path, tmp_path: Path) -> None:
    config = always_config(config_dir)
    email = FakeEmail()
    result = run(config, tmp_path, email=email)
    assert sorted(j.symbol for j in result.report.signals) == sorted(JP)
    assert result.sent == ["email"]
    assert "シグナル5件" in email.sent[0][0]
    notified = load_notified(tmp_path)
    assert notified == {notified_key(s, "always"): DAY for s in JP}
    rows = read_csv(tmp_path / "results/2026-10.csv")
    assert {r["notified"] for r in rows} == {"true"}


def test_holiday_exits_normally(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    email = FakeEmail()
    result = run(config, tmp_path, email=email, override=dt.date(2026, 10, 31))
    assert result.status == "skipped_holiday"
    assert [r["status"] for r in read_csv(tmp_path / "runs.csv")] == ["skipped_holiday"]
    assert email.sent == [] and not (tmp_path / "results").exists()


def test_target_date_can_be_given(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    src = DictSource({k: frame(i, end=dt.date(2026, 10, 29)) for i, k in enumerate(JP + ["1306"])})
    result = run(config, tmp_path, src, override=dt.date(2026, 10, 29))
    assert result.target.date == dt.date(2026, 10, 29) and result.status == "success"


# --- dry-run ------------------------------------------------------------------------


def test_dry_run_has_no_side_effects(config_dir: Path, tmp_path: Path) -> None:
    config = always_config(config_dir)
    email = FakeEmail()
    state = tmp_path / "state"
    result = run(config, state, email=email, dry_run=True)
    assert len(result.report.signals) == 5
    assert len(result.result_rows) == 5
    assert email.sent == []
    assert not state.exists()


def test_dry_run_on_holiday_writes_nothing(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    run(config, tmp_path, override=dt.date(2026, 10, 31), dry_run=True)
    assert list(tmp_path.iterdir()) == []


# --- 銘柄単位の失敗は続行、手順単位の失敗は止める ---------------------------------------


def test_partial_failure_continues_with_warning(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    email = FakeEmail()
    stale = source()
    stale.frames["8306"] = frame(2, end=dt.date(2026, 10, 29))
    stale.fail = {"6758"}
    result = run(config, tmp_path, stale, email)
    assert result.status == "partial"
    assert {r["symbol"] for r in read_csv(tmp_path / "results/2026-10.csv")} == {
        "7203",
        "8035",
        "9984",
    }
    [run_row] = read_csv(tmp_path / "runs.csv")
    assert (run_row["status"], run_row["symbols_ok"], run_row["symbols_failed"]) == (
        "partial",
        "3",
        "2",
    )
    body = email.sent[0][1]
    assert "6758 ソニーグループ: 取得に失敗しました: HTTP 500（判定から外しました）" in body
    assert "8306 三菱UFJフィナンシャル・グループ: 2026-10-30 のデータがありません" in body


def test_warnings_without_exclusion_keep_success(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    src = source()
    df = src.frames["7203"].copy()
    df.loc[df.index[-50], ["open", "high", "low", "close", "volume"]] = None  # 過去の売買なしの日
    src.frames["7203"] = df
    src.frames["6758"] = src.frames["6758"].drop(index=src.frames["6758"].index[-30])  # 歯抜け
    result = run(config, tmp_path, src)
    assert result.status == "success"
    assert {r["symbol"] for r in read_csv(tmp_path / "results/2026-10.csv")} == set(JP)
    assert any(w.startswith("7203 トヨタ自動車: 売買なしの日が1日") for w in result.report.warnings)
    assert any(
        w.startswith("6758 ソニーグループ: 立会日の行が1日ありません")
        for w in result.report.warnings
    )


def test_benchmark_failure_is_a_warning(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    result = run(config, tmp_path, source(fail={"1306"}))
    assert result.status == "partial"
    assert any(w.startswith("ベンチマーク 1306") for w in result.report.warnings)


def test_all_symbols_failing_stops(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    with pytest.raises(DailyError, match="全銘柄"):
        run(config, tmp_path, source(fail=set(JP)))
    assert read_csv(tmp_path / "runs.csv") == []


class UnavailableSource:
    def fetch(self, symbol, start, end):
        raise SourceUnavailable("J-Quants の認証ができません")


def test_unavailable_source_stops_the_run(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    with pytest.raises(SourceUnavailable):
        run(config, tmp_path, UnavailableSource())
    assert read_csv(tmp_path / "runs.csv") == []


def test_feature_errors_are_not_swallowed(tmp_path: Path, monkeypatch) -> None:
    config = load_config(ROOT / "config")

    def broken(*a, **kw):
        raise RuntimeError("bug in features")

    monkeypatch.setattr("swing_scanner.daily.compute_features", broken)
    email = FakeEmail()
    with pytest.raises(RuntimeError):
        run(config, tmp_path, email=email)
    assert email.sent == [] and list(tmp_path.iterdir()) == []


def test_send_failure_writes_nothing(config_dir: Path, tmp_path: Path) -> None:
    config = always_config(config_dir)
    state = tmp_path / "state"
    with pytest.raises(NotifyError):
        run(config, state, email=FakeEmail(fail=True))
    assert not state.exists()
    # 再実行で送り直せる
    email = FakeEmail()
    run(config, state, email=email)
    assert "シグナル5件" in email.sent[0][0]


# --- 再実行の冪等性 ------------------------------------------------------------------


def test_rerun_does_not_duplicate_logs_or_notifications(config_dir: Path, tmp_path: Path) -> None:
    config = always_config(config_dir)
    first, second = FakeEmail(), FakeEmail()
    run(config, tmp_path, email=first)
    result = run(config, tmp_path, email=second)
    assert len(read_csv(tmp_path / "results/2026-10.csv")) == 5
    assert result.report.signals == []  # 送信済みはクールダウンで弾く
    assert second.sent == []  # 通知そのものも送り直さない
    # 送ったことは判定ログに残ったまま
    assert {r["notified"] for r in read_csv(tmp_path / "results/2026-10.csv")} == {"true"}


def test_force_resends_same_day(config_dir: Path, tmp_path: Path) -> None:
    config = always_config(config_dir)
    run(config, tmp_path, email=FakeEmail())
    again = FakeEmail()
    result = run(config, tmp_path, email=again, force=True)
    assert len(result.report.signals) == 5
    assert "シグナル5件" in again.sent[0][0]


def test_rerun_without_signals_does_not_resend_empty_report(tmp_path: Path) -> None:
    config = load_config(ROOT / "config")
    run(config, tmp_path, email=FakeEmail())
    again = FakeEmail()
    run(config, tmp_path, email=again)
    assert again.sent == []
    forced = FakeEmail()
    run(config, tmp_path, email=forced, force=True)
    assert len(forced.sent) == 1


def test_cooldown_spans_days(config_dir: Path, tmp_path: Path) -> None:
    config = always_config(config_dir)
    run(config, tmp_path, email=FakeEmail())
    monday = dt.date(2026, 11, 2)
    src = DictSource({k: frame(i, end=monday) for i, k in enumerate(JP + ["1306"])})
    result = run(config, tmp_path, src, override=monday)
    assert result.report.signals == []  # 金曜に送った分は5営業日たつまで送らない


# --- CLI ---------------------------------------------------------------------------


def test_cli_dry_run_with_csv(tmp_path: Path, capsys) -> None:
    data = tmp_path / "csv"
    data.mkdir()
    for i, code in enumerate(JP + ["1306"]):
        frame(i).to_csv(data / f"{code}.csv", index=False)
    state = tmp_path / "state"
    args = ["run", "--market", "JP", "--date", "2026-10-30", "--calendar", "weekdays"]
    code = main([*args, "--data-dir", str(data), "--state-dir", str(state), "--dry-run"])
    assert code == 0
    assert "件名: 【JP】2026-10-30" in capsys.readouterr().out
    assert not state.exists()


def test_cli_exit_codes(tmp_path: Path, capsys, monkeypatch) -> None:
    for name in ("JQUANTS_API_KEY", "JQUANTS_REFRESH_TOKEN", "JQUANTS_MAIL", "JQUANTS_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    base = ["run", "--market", "JP", "--calendar", "weekdays", "--state-dir", str(tmp_path)]
    assert main([*base, "--date", "2026-10-31"]) == 0  # 休場日は API を使わない
    assert main([*base, "--date", "2026-10-30"]) == 1  # 認証情報が無い
    assert "J-Quants の認証ができません" in capsys.readouterr().err
