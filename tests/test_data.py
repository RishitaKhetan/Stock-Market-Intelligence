"""Tests for src/data.py (and, for now, a sanity check on the config)."""

from src import config


def test_config_values():
    assert config.HORIZON_DAYS == 5
