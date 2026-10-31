"""GitHub Actions の日次ワークフロー（[1-9]）が、コードと食い違っていないこと。"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/daily.yml").read_text(encoding="utf-8"))
# PyYAML は on: を True として読む
TRIGGERS = WORKFLOW[True]
JOB = WORKFLOW["jobs"]["scan"]
RUN_STEP = next(s for s in JOB["steps"] if s.get("name") == "Run daily scan")

# コードが環境変数から読む認証情報
CREDENTIALS = [
    "JQUANTS_API_KEY",
    "JQUANTS_REFRESH_TOKEN",
    "JQUANTS_MAIL",
    "JQUANTS_PASSWORD",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "MAIL_USER",
    "MAIL_APP_PASSWORD",
    "MAIL_TO",
    "LINE_CHANNEL_ACCESS_TOKEN",
    "LINE_TO",
]


def test_credentials_are_read_by_code() -> None:
    sources = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    for name in CREDENTIALS:
        assert f'"{name}"' in sources, name


def test_credentials_come_from_secrets() -> None:
    env = RUN_STEP["env"]
    assert set(env) == set(CREDENTIALS)
    for name, value in env.items():
        assert value == f"${{{{ secrets.{name} }}}}", name


def test_each_schedule_maps_to_a_market() -> None:
    crons = [s["cron"] for s in TRIGGERS["schedule"]]
    assert crons == ["17 10 * * 1-5", "43 22 * * 1-5"]
    market = JOB["env"]["MARKET"]
    # 米国株の cron だけを名指しし、それ以外（日本株の cron）は JP にする
    assert "github.event.schedule == '43 22 * * 1-5' && 'US' || 'JP'" in market
    for cron in crons:
        minute = cron.split()[0]
        assert minute not in ("0", "00", "30"), "正時・30分は混雑して遅れやすい"


def test_manual_run_inputs_reach_the_cli_through_env() -> None:
    inputs = TRIGGERS["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"market", "date", "force", "dry_run"}
    assert inputs["market"]["options"] == ["JP", "US"]
    script = RUN_STEP["run"]
    # 入力値をスクリプトに直接埋め込まない（シェルへの注入を防ぐ）
    assert "${{" not in script
    for flag in ("--market", "--date", "--force", "--dry-run"):
        assert flag in script


def test_runs_are_serialized_and_state_is_committed() -> None:
    assert WORKFLOW["concurrency"] == {"group": "state", "cancel-in-progress": False}
    assert WORKFLOW["permissions"] == {"contents": "write"}
    commit = next(s for s in JOB["steps"] if s.get("name") == "Commit state")
    assert commit["if"] == "success() && env.DRY_RUN != 'true'"
    assert "git add state/" in commit["run"]
    assert re.search(r"git pull --rebase", commit["run"])


def test_scheduled_runs_can_be_switched_off() -> None:
    assert JOB["if"] == "github.event_name == 'workflow_dispatch' || vars.DAILY_ENABLED == 'true'"
