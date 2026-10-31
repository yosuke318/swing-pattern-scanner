"""Email / LINE への送信（基本設計書 5.2 の手順10）。

- Email: Gmail の SMTP（smtp.gmail.com:465、SSL）にアプリ パスワードで認証して送る
  （market-sentiment-monitor と同じ方式）。環境変数 MAIL_USER / MAIL_APP_PASSWORD / MAIL_TO
  （MAIL_TO が無ければ自分宛て）
- LINE: Messaging API の push メッセージ。環境変数 LINE_CHANNEL_ACCESS_TOKEN / LINE_TO（送り先の
  ユーザー ID）

送信に失敗したら例外を上げる（呼び出し側で異常終了にし、notified.json を更新しない）。
有効にしたチャネルの認証情報が無いときも例外にする（黙って送らないと、届かないことに気づけない）。
"""

from __future__ import annotations

import os
import smtplib
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

from swing_scanner.config import NotificationSettings
from swing_scanner.notify.message import Report, email_body, line_text, subject
from swing_scanner.sources.http import ApiError, request_json

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"


class NotifyError(Exception):
    """通知の送信の失敗。"""


@dataclass(frozen=True)
class EmailSender:
    user: str
    app_password: str
    to: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> EmailSender:
        user = env.get("MAIL_USER", "").strip()
        # Google の画面からコピーすると区切りの空白（NBSP を含む）が混ざるので除く
        app_password = "".join(env.get("MAIL_APP_PASSWORD", "").split())
        if not (user and app_password):
            raise NotifyError(
                "メールの認証情報が環境変数にありません（MAIL_USER / MAIL_APP_PASSWORD）"
            )
        return cls(user, app_password, env.get("MAIL_TO") or user)

    def send(self, subject_text: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject_text
        msg["From"] = self.user
        msg["To"] = self.to
        msg.set_content(body)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
                smtp.login(self.user, self.app_password)
                smtp.send_message(msg)
        except (OSError, smtplib.SMTPException) as e:
            raise NotifyError(f"メールを送信できません: {e}") from e


@dataclass(frozen=True)
class LineSender:
    token: str
    to: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> LineSender:
        token, to = env.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip(), env.get("LINE_TO", "").strip()
        if not (token and to):
            raise NotifyError(
                "LINE の認証情報が環境変数にありません（LINE_CHANNEL_ACCESS_TOKEN / LINE_TO）"
            )
        return cls(token, to)

    def send(self, text: str) -> None:
        try:
            request_json(
                LINE_PUSH_URL,
                method="POST",
                headers={"Authorization": f"Bearer {self.token}"},
                body={"to": self.to, "messages": [{"type": "text", "text": text}]},
            )
        except ApiError as e:
            raise NotifyError(f"LINE に送信できません: {e}") from e


class EmailChannel(Protocol):
    def send(self, subject_text: str, body: str) -> None: ...


class LineChannel(Protocol):
    def send(self, text: str) -> None: ...


def deliver(
    report: Report,
    settings: NotificationSettings,
    email: EmailChannel | None = None,
    line: LineChannel | None = None,
) -> list[str]:
    """設定に従って送る。送ったチャネル名のリストを返す。失敗したら NotifyError。

    email / line を省略すると環境変数から作る（有効なチャネルだけ）。
    メールは該当なしの日も送る。LINE は send_when_empty でなければシグナルがある日だけ送る。
    """
    sent: list[str] = []
    if settings.email.enabled:
        (email or EmailSender.from_env()).send(subject(report), email_body(report))
        sent.append("email")
    if settings.line.enabled and (report.signals or settings.line.send_when_empty):
        (line or LineSender.from_env()).send(line_text(report))
        sent.append("line")
    return sent
