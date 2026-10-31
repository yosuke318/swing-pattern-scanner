import datetime as dt
import math
import shutil
from pathlib import Path

import pytest
import yaml

from swing_scanner.config import Config, load_config
from swing_scanner.daily import send_report
from swing_scanner.filters import FilterOutcome
from swing_scanner.notify import send as send_module
from swing_scanner.notify.cooldown import (
    business_days_between,
    in_cooldown,
    prune,
    record_notified,
)
from swing_scanner.notify.message import (
    LINE_MAX_CHARS,
    Report,
    email_body,
    format_price,
    format_value,
    line_text,
    subject,
)
from swing_scanner.notify.select import Judgement, select
from swing_scanner.notify.send import EmailSender, LineSender, NotifyError, deliver
from swing_scanner.rules.engine import evaluate
from swing_scanner.sources.http import ApiError
from swing_scanner.state import load_notified, notified_key, save_notified

ROOT = Path(__file__).resolve().parents[1]
D = dt.date
MON, TUE, FRI, NEXT_MON = D(2026, 11, 2), D(2026, 11, 3), D(2026, 11, 6), D(2026, 11, 9)
NO_FILTER = FilterOutcome((), ())


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "config", tmp_path / "config")
    return tmp_path / "config"


def make_config(config_dir: Path, **notification) -> Config:
    path = config_dir / "settings.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in notification.items():
        if isinstance(value, dict):
            data["notification"][key].update(value)
        else:
            data["notification"][key] = value
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(config_dir)


CONFIG = load_config(ROOT / "config")
PATTERN = CONFIG.pattern("golden_cross_uptrend")


def row(**overrides) -> dict:
    base = {
        "close": 1000.0,
        "atr14": 20.0,
        "swing_low_price": 980.0,
        "days_since_golden_cross": 3.0,
        "swing_low_higher_than_prior": True,
        "adx14": 25.0,
        "close_minus_ma25_atr_ratio": 0.2,
        "vol_regime": "normal",
        "trend_regime": "uptrend",
    }
    return base | overrides


def judgement(symbol: str, outcome: FilterOutcome = NO_FILTER, pattern=PATTERN, **overrides):
    return Judgement(
        "JP", symbol, f"銘柄{symbol}", pattern, evaluate(pattern, row(**overrides)), outcome
    )


# --- クールダウン ------------------------------------------------------------------


def test_business_days_between() -> None:
    assert business_days_between(MON, NEXT_MON) == 5
    sessions = [MON, D(2026, 11, 4), D(2026, 11, 5), FRI, NEXT_MON]  # 11/3 は祝日
    assert business_days_between(MON, NEXT_MON, sessions) == 4


def test_in_cooldown_counts_business_days() -> None:
    notified = {notified_key("7203", "p"): MON}
    assert in_cooldown(notified, "7203", "p", FRI, 5)  # 4営業日
    assert not in_cooldown(notified, "7203", "p", NEXT_MON, 5)  # 5営業日
    assert not in_cooldown(notified, "7203", "other", FRI, 5)
    assert not in_cooldown(notified, "7203", "p", FRI, 0)


def test_same_day_resend_only_with_force() -> None:
    notified = {notified_key("7203", "p"): MON}
    assert in_cooldown(notified, "7203", "p", MON, 5)
    assert not in_cooldown(notified, "7203", "p", MON, 5, resend_same_day=True)
    assert in_cooldown(notified, "7203", "p", TUE, 5, resend_same_day=True)


def test_record_and_prune() -> None:
    old = {notified_key("7203", "p"): D(2026, 10, 26), notified_key("6758", "p"): D(2026, 10, 30)}
    updated = record_notified(old, [("8035", "p")], MON)
    assert updated[notified_key("8035", "p")] == MON
    assert old == {
        notified_key("7203", "p"): D(2026, 10, 26),
        notified_key("6758", "p"): D(2026, 10, 30),
    }
    # 10/26 から 11/2 は5営業日でクールダウン明け。10/30 からは1営業日
    assert set(prune(updated, MON, 5)) == {notified_key("6758", "p"), notified_key("8035", "p")}


# --- 通知対象の抽出 ----------------------------------------------------------------


def test_select_signals_and_near_misses() -> None:
    judgements = [
        judgement("1001"),  # 100点
        judgement("1002", adx14=15.0),  # 75点（しきい値ちょうど）→ シグナル
        judgement("1003", adx14=15.0, swing_low_higher_than_prior=False),  # 50点 → 惜しい
        judgement(
            "1004", adx14=15.0, swing_low_higher_than_prior=False, days_since_golden_cross=9.0
        ),  # 25点
        judgement("1005", vol_regime="high"),  # レジーム対象外
        judgement("1006", FilterOutcome(("決算: …",), ())),  # 除外
        judgement(
            "1007", FilterOutcome(("流動性: …",), ()), adx14=15.0, swing_low_higher_than_prior=False
        ),
        judgement("1008", atr14=math.nan, close_minus_ma25_atr_ratio=math.nan),  # 出口なし
    ]
    result = select(judgements, CONFIG, {}, MON)
    assert [j.symbol for j in result.signals] == ["1001", "1002"]
    assert [j.symbol for j in result.near_misses] == ["1003"]


def test_select_skips_cooldown(config_dir: Path) -> None:
    notified = {notified_key("1001", PATTERN.pattern_id): FRI}
    result = select([judgement("1001"), judgement("1002")], CONFIG, notified, NEXT_MON)
    assert [j.symbol for j in result.signals] == ["1002"]
    forced = select(
        [judgement("1001")],
        CONFIG,
        {notified_key("1001", PATTERN.pattern_id): MON},
        MON,
        resend_same_day=True,
    )
    assert [j.symbol for j in forced.signals] == ["1001"]


def test_threshold_and_near_miss_come_from_settings(config_dir: Path) -> None:
    path = config_dir / "patterns/golden_cross_uptrend/v1.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["score_threshold"] = 100
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = make_config(config_dir, near_miss={"min_score": 75, "max_items": 1})
    pattern = config.pattern("golden_cross_uptrend")
    js = [
        judgement("1001", pattern=pattern),
        judgement("1002", pattern=pattern, adx14=15.0),
        judgement("1003", pattern=pattern, swing_low_higher_than_prior=False),
    ]
    result = select(js, config, {}, MON)
    assert [j.symbol for j in result.signals] == ["1001"]
    assert [j.symbol for j in result.near_misses] == ["1002"]  # 75点が2件、上限1件


# --- 通知文 ------------------------------------------------------------------------


def report(**kw) -> Report:
    base = dict(market="JP", target_date=MON, signals=[], thresholds={PATTERN.pattern_id: 75})
    return Report(**(base | kw))


def test_subject_variants() -> None:
    assert subject(report()) == "【JP】2026-11-02 該当なし"
    assert subject(report(warnings=["x"])) == "【JP】2026-11-02 該当なし / 警告1件"
    two = [judgement("1001"), judgement("1002")]
    assert subject(report(signals=two, warnings=["x"])) == "【JP】2026-11-02 シグナル2件 / 警告1件"


def test_email_body_has_what_is_needed_to_decide() -> None:
    signal = judgement("7203", FilterOutcome((), ("市場環境: 指数が downtrend",)), adx14=15.0)
    near = judgement("8035", adx14=15.0, swing_low_higher_than_prior=False)
    body = email_body(report(signals=[signal], near_misses=[near], warnings=["6758: 取得に失敗"]))
    assert "2026-11-02（月）の日本株の判定結果です。" in body
    assert "このツールは発注しません" in body
    assert "1. 7203 銘柄7203 — ゴールデンクロス＋上昇構造（v1）" in body
    assert "スコア 75/100（しきい値 75）" in body
    assert "○ ゴールデンクロスからの営業日数: 3（基準 <= 5）" in body
    assert "× ADX14（トレンドの強さ）: 15（基準 >= 20）" in body
    assert "○ (終値 − MA25) ÷ ATR14: 0.2（基準 -1〜0.5）" in body
    assert "エントリーの目安: 翌営業日の寄り付き" in body
    assert "損切りの目安: 980（-2.0%、直近スイング安値）" in body
    assert "利確の目安: 1,060（+6.0%）" in body
    assert "最大保有: 10営業日" in body
    assert "注意: 市場環境: 指数が downtrend" in body
    assert "- 8035 銘柄8035 — ゴールデンクロス＋上昇構造 スコア 50/100（しきい値 75）" in body
    assert " — 未達: 直近の確定スイング安値 > その前の安値、ADX14（トレンドの強さ）" in body
    assert "○ 直近の確定スイング安値 > その前の安値: はい（基準 はい）" in body
    assert "■ 警告 1件\n- 6758: 取得に失敗" in body


def test_email_body_for_empty_day() -> None:
    body = email_body(report())
    assert "■ シグナル: 該当なし" in body
    assert "■ 警告: なし" in body
    assert "惜しかった" not in body


def test_gap_skip_price_is_shown() -> None:
    data = PATTERN.model_dump()
    data["exit"]["skip_gap_atr_mult"] = 1.0
    gapped = PATTERN.model_validate(data)
    body = email_body(report(signals=[judgement("7203", pattern=gapped)]))
    assert "翌営業日の寄り付き（1,020 以上の窓開けで寄り付いたら見送り）" in body


def test_unevaluable_condition_is_shown() -> None:
    near = judgement("8035", adx14=math.nan, swing_low_higher_than_prior=False)
    assert "未達: 直近の確定スイング安値 > その前の安値、ADX14（トレンドの強さ）" in email_body(
        report(near_misses=[near])
    )
    signal = judgement("7203", adx14=math.nan)
    assert "－ ADX14（トレンドの強さ）: 評価不能（基準 >= 20）" in email_body(
        report(signals=[signal])
    )


def test_line_text_is_short() -> None:
    text = line_text(
        report(
            signals=[judgement("7203")],
            near_misses=[judgement("8035", adx14=15.0, swing_low_higher_than_prior=False)],
        )
    )
    assert text.splitlines()[:4] == [
        "【JP】2026-11-02 シグナル1件",
        "",
        "7203 銘柄7203 100点",
        "損切り 980（-2.0%） / 利確 1,060（+6.0%） / 10日",
    ]
    assert "惜しい: 8035（50点）" in text
    assert "根拠" in text and "○" not in text


def test_line_text_is_truncated() -> None:
    many = [judgement(f"{1000 + i}") for i in range(200)]
    text = line_text(report(signals=many))
    assert len(text) <= LINE_MAX_CHARS
    assert text.endswith("続きはメールで）")


@pytest.mark.parametrize(
    "value, market, expected",
    [(1234.0, "JP", "1,234"), (1234.5, "JP", "1,234.5"), (123.456, "US", "123.46")],
)
def test_format_price(value, market, expected) -> None:
    assert format_price(value, market) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "評価不能"),
        (True, "はい"),
        (math.inf, "なし"),
        (3.0, "3"),
        (0.25, "0.25"),
        (25.123, "25.12"),
        (5e9, "5,000,000,000"),
    ],
)
def test_format_value(value, expected) -> None:
    assert format_value(value) == expected


# --- 送信 --------------------------------------------------------------------------


class FakeEmail:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def send(self, subject_text: str, body: str) -> None:
        if self.fail:
            raise NotifyError("SMTP down")
        self.sent.append((subject_text, body))


class FakeLine:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)


def test_email_sender_from_env() -> None:
    sender = EmailSender.from_env(
        {"MAIL_USER": " me@example.com ", "MAIL_APP_PASSWORD": "abcd\xa0efgh ijkl"}
    )
    assert (sender.user, sender.app_password, sender.to) == (
        "me@example.com",
        "abcdefghijkl",
        "me@example.com",
    )
    with pytest.raises(NotifyError, match="MAIL_USER"):
        EmailSender.from_env({})


def test_email_sender_uses_gmail_ssl(monkeypatch) -> None:
    calls = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            calls["login"] = (user, password)

        def send_message(self, msg):
            calls["msg"] = msg

    monkeypatch.setattr(send_module.smtplib, "SMTP_SSL", FakeSMTP)
    EmailSender("me@example.com", "pw", "you@example.com").send("件名", "本文")
    assert calls["server"] == ("smtp.gmail.com", 465)
    assert calls["login"] == ("me@example.com", "pw")
    assert calls["msg"]["To"] == "you@example.com"
    assert calls["msg"].get_content().strip() == "本文"


def test_line_sender(monkeypatch) -> None:
    calls = {}

    def fake_request_json(url, method="GET", headers=None, body=None, **kw):
        calls.update(url=url, method=method, headers=headers, body=body)
        return {}

    monkeypatch.setattr(send_module, "request_json", fake_request_json)
    LineSender.from_env({"LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_TO": "U123"}).send("hello")
    assert calls["url"] == "https://api.line.me/v2/bot/message/push"
    assert calls["headers"] == {"Authorization": "Bearer t"}
    assert calls["body"] == {"to": "U123", "messages": [{"type": "text", "text": "hello"}]}

    def failing(*a, **kw):
        raise ApiError("HTTP 401")

    monkeypatch.setattr(send_module, "request_json", failing)
    with pytest.raises(NotifyError, match="HTTP 401"):
        LineSender("t", "U123").send("hello")
    with pytest.raises(NotifyError, match="LINE_TO"):
        LineSender.from_env({"LINE_CHANNEL_ACCESS_TOKEN": "t"})


def test_deliver_channels(config_dir: Path) -> None:
    both = make_config(config_dir, line={"enabled": True}).settings.notification
    email, line = FakeEmail(), FakeLine()
    assert deliver(report(), both, email, line) == ["email"]  # 該当なしの日は LINE に送らない
    assert deliver(report(signals=[judgement("7203")]), both, email, line) == ["email", "line"]
    assert len(email.sent) == 2 and len(line.sent) == 1

    always = make_config(config_dir, line={"send_when_empty": True}).settings.notification
    assert deliver(report(), always, FakeEmail(), FakeLine()) == ["email", "line"]

    off = make_config(config_dir, email={"enabled": False}, line={"enabled": False})
    assert deliver(report(), off.settings.notification, FakeEmail(), FakeLine()) == []


def test_enabled_channel_without_credentials_fails(monkeypatch) -> None:
    monkeypatch.delenv("MAIL_USER", raising=False)
    with pytest.raises(NotifyError):
        deliver(report(), CONFIG.settings.notification)


# --- 送信と記録の順序 --------------------------------------------------------------


def test_send_report_records_after_sending(tmp_path: Path) -> None:
    save_notified(tmp_path, {notified_key("9999", "p"): D(2026, 10, 1)})
    sent = send_report(report(signals=[judgement("7203")]), CONFIG, tmp_path, email=FakeEmail())
    assert sent == ["email"]
    # 送った銘柄が記録され、クールダウンを過ぎた古いエントリは消える
    assert load_notified(tmp_path) == {notified_key("7203", PATTERN.pattern_id): MON}


def test_failed_send_does_not_update_notified(tmp_path: Path) -> None:
    before = {notified_key("6758", PATTERN.pattern_id): D(2026, 10, 30)}
    save_notified(tmp_path, before)
    with pytest.raises(NotifyError):
        send_report(
            report(signals=[judgement("7203")]), CONFIG, tmp_path, email=FakeEmail(fail=True)
        )
    assert load_notified(tmp_path) == before
    # 再実行すれば送り直せる（クールダウンに入っていない）
    assert select([judgement("7203")], CONFIG, load_notified(tmp_path), MON).signals
