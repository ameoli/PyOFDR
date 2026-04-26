"""Smoke tests for the CLI (#29)."""

import pytest
import yaml

from helpers import CFG
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


def _write_run_cfg(tmp_path, output_path=None):
    """small CFG dumped to YAML, optionally with output.path set."""
    cfg = {**CFG}
    if output_path is not None:
        cfg = {**cfg, "output": {"path": str(output_path)}}
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_run_writes_hdf5_when_output_set(tmp_path):
    out = tmp_path / "out.h5"
    cfg_path = _write_run_cfg(tmp_path, output_path=out)
    code = main(["run", str(cfg_path)])
    assert code == 0
    assert out.exists()


def test_run_no_output_path_still_succeeds(tmp_path):
    # no output.path -> nothing written, but exit 0 is fine
    cfg_path = _write_run_cfg(tmp_path)
    code = main(["run", str(cfg_path)])
    assert code == 0


def test_run_missing_config_exits_2(capsys):
    code = main(["run", "configs/does_not_exist.yaml"])
    assert code == 2


def test_run_output_flag_overrides_config(tmp_path):
    # config has no output.path, -o picks one
    cfg_path = _write_run_cfg(tmp_path)
    out = tmp_path / "via_flag.h5"
    code = main(["run", str(cfg_path), "-o", str(out)])
    assert code == 0
    assert out.exists()


def test_run_output_flag_wins_over_config(tmp_path):
    # config sets a path, -o should win
    in_cfg = tmp_path / "from_cfg.h5"
    cfg_path = _write_run_cfg(tmp_path, output_path=in_cfg)
    out = tmp_path / "from_flag.h5"
    code = main(["run", str(cfg_path), "-o", str(out)])
    assert code == 0
    assert out.exists()
    assert not in_cfg.exists()


def test_run_quiet_sets_root_logger_to_warning(tmp_path):
    import logging
    cfg_path = _write_run_cfg(tmp_path)
    code = main(["run", str(cfg_path), "-q"])
    assert code == 0
    assert logging.getLogger().level == logging.WARNING


def test_run_default_logs_at_info(tmp_path):
    import logging
    cfg_path = _write_run_cfg(tmp_path)
    code = main(["run", str(cfg_path)])
    assert code == 0
    assert logging.getLogger().level == logging.INFO
