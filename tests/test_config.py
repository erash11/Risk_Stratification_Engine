from pathlib import Path

import pytest

from risk_stratification_engine.config import (
    DATA_SOURCE_PATH_KEYS,
    DataSourcePaths,
    load_data_source_paths,
)


def test_data_source_path_keys_match_live_sources():
    assert DATA_SOURCE_PATH_KEYS == (
        "forceplate_db",
        "gps_db",
        "bodyweight_csv",
        "perch_db",
        "injury_csv",
    )


def test_load_data_source_paths_reads_yaml_without_requiring_files(tmp_path):
    config_path = tmp_path / "paths.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"forceplate_db: {tmp_path / 'forceplate.db'}",
                f"gps_db: {tmp_path / 'gps_history.duckdb'}",
                f"bodyweight_csv: {tmp_path / 'BodyWeightMaster.csv'}",
                f"perch_db: {tmp_path / 'perch.duckdb'}",
                f"injury_csv: {tmp_path / 'injuries.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    paths = load_data_source_paths(config_path, require_exists=False)

    assert isinstance(paths, DataSourcePaths)
    assert paths.forceplate_db == tmp_path / "forceplate.db"
    assert paths.gps_db == tmp_path / "gps_history.duckdb"
    assert paths.bodyweight_csv == tmp_path / "BodyWeightMaster.csv"
    assert paths.perch_db == tmp_path / "perch.duckdb"
    assert paths.injury_csv == tmp_path / "injuries.csv"


def test_load_data_source_paths_rejects_missing_required_key(tmp_path):
    config_path = tmp_path / "paths.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"forceplate_db: {tmp_path / 'forceplate.db'}",
                f"gps_db: {tmp_path / 'gps_history.duckdb'}",
                f"bodyweight_csv: {tmp_path / 'BodyWeightMaster.csv'}",
                f"perch_db: {tmp_path / 'perch.duckdb'}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_data_source_paths(config_path, require_exists=False)

    assert "paths config missing required keys: injury_csv" in str(exc.value)


def test_load_data_source_paths_rejects_unknown_key(tmp_path):
    config_path = tmp_path / "paths.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"forceplate_db: {tmp_path / 'forceplate.db'}",
                f"gps_db: {tmp_path / 'gps_history.duckdb'}",
                f"bodyweight_csv: {tmp_path / 'BodyWeightMaster.csv'}",
                f"perch_db: {tmp_path / 'perch.duckdb'}",
                f"injury_csv: {tmp_path / 'injuries.csv'}",
                "extra_db: C:/tmp/extra.duckdb",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        load_data_source_paths(config_path, require_exists=False)

    assert "paths config contains unknown keys: extra_db" in str(exc.value)


def test_load_data_source_paths_reports_missing_files(tmp_path):
    existing_paths = {
        "forceplate_db": tmp_path / "forceplate.db",
        "gps_db": tmp_path / "gps_history.duckdb",
        "bodyweight_csv": tmp_path / "BodyWeightMaster.csv",
        "perch_db": tmp_path / "perch.duckdb",
    }
    for path in existing_paths.values():
        path.write_text("", encoding="utf-8")

    config_path = tmp_path / "paths.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"forceplate_db: {existing_paths['forceplate_db']}",
                f"gps_db: {existing_paths['gps_db']}",
                f"bodyweight_csv: {existing_paths['bodyweight_csv']}",
                f"perch_db: {existing_paths['perch_db']}",
                f"injury_csv: {tmp_path / 'injuries.csv'}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError) as exc:
        load_data_source_paths(config_path)

    assert "configured data source paths do not exist" in str(exc.value)
    assert "injury_csv" in str(exc.value)
