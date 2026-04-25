from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


DATA_SOURCE_PATH_KEYS = (
    "forceplate_db",
    "gps_db",
    "bodyweight_csv",
    "perch_db",
    "injury_csv",
)

DEFAULT_PATHS_CONFIG = Path("config/paths.local.yaml")


@dataclass(frozen=True)
class DataSourcePaths:
    forceplate_db: Path
    gps_db: Path
    bodyweight_csv: Path
    perch_db: Path
    injury_csv: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "forceplate_db": self.forceplate_db,
            "gps_db": self.gps_db,
            "bodyweight_csv": self.bodyweight_csv,
            "perch_db": self.perch_db,
            "injury_csv": self.injury_csv,
        }


def load_data_source_paths(
    config_path: str | Path = DEFAULT_PATHS_CONFIG,
    *,
    require_exists: bool = True,
) -> DataSourcePaths:
    config_file = Path(config_path)
    raw_config = _read_yaml_mapping(config_file)
    _validate_config_keys(raw_config)

    paths = DataSourcePaths(
        forceplate_db=_normalize_path(raw_config["forceplate_db"]),
        gps_db=_normalize_path(raw_config["gps_db"]),
        bodyweight_csv=_normalize_path(raw_config["bodyweight_csv"]),
        perch_db=_normalize_path(raw_config["perch_db"]),
        injury_csv=_normalize_path(raw_config["injury_csv"]),
    )
    if require_exists:
        _require_existing_paths(paths)
    return paths


def _read_yaml_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"paths config not found: {config_path}. "
            "Copy config/paths.example.yaml to config/paths.local.yaml."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("paths config must be a YAML mapping")
    return loaded


def _validate_config_keys(config: dict[str, Any]) -> None:
    provided_keys = set(config)
    expected_keys = set(DATA_SOURCE_PATH_KEYS)
    missing = sorted(expected_keys - provided_keys)
    unknown = sorted(provided_keys - expected_keys)
    if missing:
        raise ValueError(f"paths config missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"paths config contains unknown keys: {', '.join(unknown)}")


def _normalize_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured data source paths must be nonblank strings")
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    return Path(expanded)


def _require_existing_paths(paths: DataSourcePaths) -> None:
    missing = [
        f"{name}={path}"
        for name, path in paths.as_dict().items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "configured data source paths do not exist: " + "; ".join(missing)
        )
