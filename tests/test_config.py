import shutil
from pathlib import Path

import pytest
import yaml

from swing_scanner.config import ConfigError, load_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, dst)
    return dst


def _edit(path: Path, fn) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_repo_config_is_valid():
    config = load_config(REPO_CONFIG)
    assert config.settings.fetch.lookback_days == 300
    assert config.active_patterns()


def test_default_score_threshold_is_used_when_pattern_omits_it(config_dir: Path):
    path = config_dir / "patterns/golden_cross_uptrend/v1.yaml"
    _edit(path, lambda d: d.pop("score_threshold"))
    config = load_config(config_dir)
    assert config.score_threshold(config.patterns[0]) == 75


@pytest.mark.parametrize(
    ("relpath", "mutate", "message"),
    [
        ("settings.yaml", lambda d: d["fetch"].pop("lookback_days"), "lookback_days"),
        ("settings.yaml", lambda d: d.update(unknown_key=1), "unknown_key"),
        ("settings.yaml", lambda d: d["fetch"].update(lookback_days="300"), "lookback_days"),
        ("settings.yaml", lambda d: d["markets"].pop("US"), "US"),
        ("settings.yaml", lambda d: d["market_filter"].update(mode="mute"), "mode"),
        ("settings.yaml", lambda d: d.pop("features"), "features"),
        ("settings.yaml", lambda d: d["features"].update(trend_method="x"), "trend_method"),
        ("settings.yaml", lambda d: d["features"].update(golden_cross_pair=[75, 25]), "短期"),
        ("symbols.yaml", lambda d: d[0].update(symbol=7203), "symbol"),
        ("symbols.yaml", lambda d: d.append(dict(d[0])), "重複"),
        ("symbols.yaml", lambda d: d[0].update(market="HK"), "market"),
        ("symbols.yaml", lambda d: d[0].update(symbol="72030"), "銘柄コード"),
        ("symbols.yaml", lambda d: d[0].update(symbol="NVDA"), "銘柄コード"),
        ("symbols.yaml", lambda d: d[-1].update(symbol="nvda"), "銘柄コード"),
        ("symbols.yaml", lambda d: d.append({**d[0], "symbol": "1306"}), "ベンチマーク"),
        (
            "settings.yaml",
            lambda d: d["markets"]["US"].update(benchmark_symbol="spy"),
            "銘柄コード",
        ),
    ],
)
def test_invalid_config_is_rejected(config_dir: Path, relpath, mutate, message):
    _edit(config_dir / relpath, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)


def test_event_for_unknown_symbol_is_rejected(config_dir: Path):
    (config_dir / "events.yaml").write_text(
        "- {symbol: '9999', market: JP, date: 2026-11-05, type: earnings}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="symbols.yaml に無い"):
        load_config(config_dir)


PATTERN = "patterns/golden_cross_uptrend/v1.yaml"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["conditions"][0].update(weight=30), "合計が 100"),
        (lambda d: d["conditions"][3].update(value=[0.5, -1.0]), "下限が上限"),
        (lambda d: d["conditions"][3].update(value=0.5), r"\[下限, 上限\]"),
        (lambda d: d["conditions"][0].update(value=[1, 2]), "リスト"),
        (lambda d: d["conditions"][1].update(op=">="), "真偽値"),
        (lambda d: d["conditions"][0].update(op="=~"), "op"),
        (lambda d: d["conditions"][1].update(key="adx14"), "重複"),
        (lambda d: d["regimes"].update(vol=["extreme"]), "vol"),
        (lambda d: d["exit"].pop("max_holding_days"), "max_holding_days"),
        (lambda d: d["exit"]["stop"].update(atr_mult=0), "atr_mult"),
        (lambda d: d.update(version=2), "ファイル名と一致"),
        (lambda d: d.update(pattern_id="other"), "ディレクトリ名と一致"),
    ],
)
def test_invalid_pattern_is_rejected(config_dir: Path, mutate, message):
    _edit(config_dir / PATTERN, mutate)
    with pytest.raises(ConfigError, match=message):
        load_config(config_dir)


def test_only_one_active_version_per_pattern(config_dir: Path):
    v1 = config_dir / PATTERN
    v2 = v1.with_name("v2.yaml")
    shutil.copy(v1, v2)
    _edit(v2, lambda d: d.update(version=2))
    with pytest.raises(ConfigError, match="複数"):
        load_config(config_dir)

    _edit(v1, lambda d: d.update(active=False))
    config = load_config(config_dir)
    assert [p.version for p in config.active_patterns()] == [2]


def test_pattern_file_name_must_be_versioned(config_dir: Path):
    shutil.copy(config_dir / PATTERN, config_dir / "patterns/golden_cross_uptrend/latest.yaml")
    with pytest.raises(ConfigError, match="v<番号>.yaml"):
        load_config(config_dir)


def test_watched_symbols_by_market(config_dir: Path):
    def mutate(d):
        d[0]["watched"] = False  # 7203 を除外
        d.append({"symbol": "130A", "name": "英字入り", "market": "JP", "watched": True})
        d.append({"symbol": "BRK.B", "name": "Berkshire B", "market": "US", "watched": True})
        # 除外済みならベンチマークと同じコードでも可
        d.append({"symbol": "SPY", "name": "SPDR S&P 500", "market": "US", "watched": False})

    _edit(config_dir / "symbols.yaml", mutate)
    config = load_config(config_dir)

    jp = [s.symbol for s in config.watched_symbols("JP")]
    us = [s.symbol for s in config.watched_symbols("US")]
    assert "7203" not in jp
    assert "130A" in jp
    assert all(s.market == "JP" for s in config.watched_symbols("JP"))
    assert "BRK.B" in us
    assert "SPY" not in us
    assert not set(jp) & set(us)


def test_repo_symbols_cover_both_markets():
    config = load_config(REPO_CONFIG)
    assert config.watched_symbols("JP")
    assert config.watched_symbols("US")
    assert config.settings.markets["JP"].benchmark_symbol == "1306"
    assert config.settings.markets["US"].benchmark_symbol == "SPY"
