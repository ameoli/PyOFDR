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


def test_validate_accepts_good_config(capsys):
    code = main(["validate", "configs/ofdr_basic.yaml"])
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_validate_missing_config_exits_2(capsys):
    code = main(["validate", "configs/does_not_exist.yaml"])
    assert code == 2


def test_validate_rejects_bad_config(tmp_path, capsys):
    # negative fiber length should fail the pydantic model
    bad = tmp_path / "bad.yaml"
    bad.write_text("fiber:\n  length: -1\n")
    code = main(["validate", str(bad)])
    assert code == 1
    assert "invalid config" in capsys.readouterr().err
