"""Smoke tests for the CLI (#29). Only `info` for now."""

import pytest

from cli import main


def test_info_prints_summary(capsys):
    code = main(["info", "configs/ofdr_basic.yaml"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Spatial resolution" in out
    assert "Nyquist" in out


def test_info_missing_config_exits_nonzero(capsys):
    code = main(["info", "configs/does_not_exist.yaml"])
    assert code == 2
    err = capsys.readouterr().err
    assert "does_not_exist.yaml" in err


def test_no_subcommand_is_an_error():
    # argparse with required=True raises SystemExit(2) here
    with pytest.raises(SystemExit):
        main([])
