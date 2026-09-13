from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def _as_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


@dataclass
class VenueConfig:
    enabled: bool = True
    max_markets: int = 40
    min_volume_24h: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Mapping:
    ticker: str
    keywords: list[str]


@dataclass
class Settings:
    root: Path = ROOT
    db_path: Path = ROOT / "data" / "quantvision.db"
    artifacts_dir: Path = ROOT / "artifacts"
    snapshot_path: Path = ROOT / "quantvisionUI" / "snapshot.json"
    website_snapshot_path: Path | None = None
    poll_seconds: int = 20
    request_timeout: float = 20.0
    kalshi: VenueConfig = field(default_factory=VenueConfig)
    polymarket: VenueConfig = field(default_factory=VenueConfig)
    default_threshold: float = 0.10
    major_threshold: float = 0.05
    major_min_volume: float = 100_000.0
    alert_log: Path = ROOT / "data" / "alerts.log"
    webhook_url: str | None = None
    web_host: str = "127.0.0.1"
    web_port: int = 8787
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://127.0.0.1:8787",
            "http://localhost:8787",
            "https://jacobhmaiddout.com",
            "https://www.jacobhmaiddout.com",
            "https://whoizjake.github.io",
        ]
    )
    mappings: list[Mapping] = field(default_factory=list)

    @property
    def kalshi_base(self) -> str:
        return self.kalshi.extra.get(
            "base_url",
            "https://api.elections.kalshi.com/trade-api/v2",
        )

    @property
    def poly_gamma(self) -> str:
        return self.polymarket.extra.get("gamma_url", "https://gamma-api.polymarket.com")

    @property
    def poly_clob(self) -> str:
        return self.polymarket.extra.get("clob_url", "https://clob.polymarket.com")

    @property
    def poly_data(self) -> str:
        return self.polymarket.extra.get("data_url", "https://data-api.polymarket.com")


def _venue(raw: dict[str, Any] | None, defaults: VenueConfig) -> VenueConfig:
    raw = raw or {}
    known = {"enabled", "max_markets", "min_volume_24h"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return VenueConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        max_markets=int(raw.get("max_markets", defaults.max_markets)),
        min_volume_24h=float(raw.get("min_volume_24h", defaults.min_volume_24h)),
        extra=extra,
    )


def load_settings(path: str | Path | None = None) -> Settings:
    root = ROOT
    config_path = Path(path) if path else Path(os.environ.get("QUANTVISION_CONFIG", DEFAULT_CONFIG))
    raw: dict[str, Any] = {}
    if config_path.exists():
        if yaml is None:
            raise RuntimeError("PyYAML is required to load config.yaml")
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {config_path} must be a mapping")
        raw = loaded

    etl = raw.get("etl") or {}
    monitor = raw.get("monitor") or {}
    web = raw.get("web") or {}
    settings = Settings()
    settings.root = root
    settings.db_path = _as_path(etl.get("db_path", settings.db_path), root)
    settings.artifacts_dir = _as_path(etl.get("artifacts_dir", settings.artifacts_dir), root)
    settings.snapshot_path = _as_path(etl.get("snapshot_path", settings.snapshot_path), root)
    if etl.get("website_snapshot_path"):
        settings.website_snapshot_path = _as_path(etl["website_snapshot_path"], root)
    settings.poll_seconds = int(etl.get("poll_seconds", settings.poll_seconds))
    settings.request_timeout = float(etl.get("request_timeout", settings.request_timeout))
    settings.kalshi = _venue(etl.get("kalshi"), VenueConfig(max_markets=40, min_volume_24h=50))
    settings.polymarket = _venue(
        etl.get("polymarket"),
        VenueConfig(max_markets=40, min_volume_24h=10_000),
    )
    settings.default_threshold = float(monitor.get("default_threshold", settings.default_threshold))
    settings.major_threshold = float(monitor.get("major_threshold", settings.major_threshold))
    settings.major_min_volume = float(monitor.get("major_min_volume", settings.major_min_volume))
    settings.alert_log = _as_path(monitor.get("alert_log", settings.alert_log), root)
    webhook = monitor.get("webhook_url") or os.environ.get("QUANTVISION_WEBHOOK_URL")
    settings.webhook_url = str(webhook) if webhook else None
    settings.web_host = str(web.get("host", settings.web_host))
    settings.web_port = int(web.get("port", settings.web_port))
    if web.get("cors_origins"):
        settings.cors_origins = list(web["cors_origins"])
    mappings = []
    for item in raw.get("mappings") or []:
        ticker = str(item.get("ticker", "")).strip()
        keywords = [str(k).lower() for k in item.get("keywords") or [] if str(k).strip()]
        if ticker and keywords:
            mappings.append(Mapping(ticker=ticker.upper(), keywords=keywords))
    if not mappings:
        mappings = [
            Mapping("BTC-USD", ["bitcoin", "btc"]),
            Mapping("ETH-USD", ["ethereum", "ether", "eth"]),
            Mapping("SPY", ["s&p", "spx", "spy", "s and p"]),
            Mapping("NVDA", ["nvidia", "nvda"]),
        ]
    settings.mappings = mappings
    return settings
