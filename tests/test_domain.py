from risk_stratification_engine import __version__
from risk_stratification_engine.cli import main


def test_package_imports():
    assert __version__ == "0.1.0"


def test_cli_main_returns_success():
    assert main([]) == 0
