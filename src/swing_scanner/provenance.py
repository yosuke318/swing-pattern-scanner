"""判定ログに残す実行コードの識別子。"""

import os
import subprocess


def commit_sha() -> str:
    """実行中のコードのコミット SHA を返す。取れなければ "unknown"。

    GitHub Actions では GITHUB_SHA、ローカルでは git rev-parse HEAD を使う。
    """
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"
