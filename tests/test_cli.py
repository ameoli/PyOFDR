"""Smoke tests for the CLI (#29)."""

import pytest
import yaml

from helpers import CFG
from pyofdr.cli import main


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


@pytest.mark.parametrize("command", ["validate", "run"])
def test_nan_strain_is_rejected_before_simulation(tmp_path, capsys, command):
    path = tmp_path / "nan.yaml"
    path.write_text("fiber: {length: 0.15}\nsource: {sweep_duration: 0.0001}\n"
                    "strain: {segments: [{start: 0, end: 0.1, epsilon: .nan}]}\n")
    output = tmp_path / "result.h5"
    args = [command, str(path)]
    if command == "run":
        args.extend(["-o", str(output)])
    assert main(args) == 1
    error = capsys.readouterr().err
    assert "invalid config" in error
    assert "strain.segments.0.epsilon" in error
    assert not output.exists()


@pytest.mark.parametrize("command", ["validate", "run"])
def test_negative_reflector_position_is_rejected(tmp_path, capsys, command):
    path = tmp_path / "reflector.yaml"
    path.write_text("fiber: {length: 0.15, reflectors: [{z: '-5 cm', R: 0.25}]}\n")
    assert main([command, str(path)]) == 1
    assert "fiber.reflectors.0.z" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["validate", "run"])
@pytest.mark.parametrize("noise", ["flicker_noise_Hz", "random_walk_noise_Hz"])
@pytest.mark.parametrize("count", [0.5, 1.0, 1.5, 2.0])
def test_colored_noise_requires_two_samples(tmp_path, capsys, command, noise, count):
    cfg = {
        "fiber": {"length": 0.01},
        "source": {"sweep_range": 1e-15, "sweep_duration": count / 1e8, noise: 1.0},
        "adc": {"sample_rate": 1e8},
    }
    path = tmp_path / "noise.yaml"
    path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "noise.h5"
    args = [command, str(path)]
    if command == "run":
        args.extend(["-o", str(output)])
    result = main(args)
    if count <= 1:
        assert result == 1
        error = capsys.readouterr().err
        assert "invalid config" in error
        assert "at least two samples" in error
        assert not output.exists()
    else:
        assert result == 0
        if command == "run":
            assert output.exists()


def _write_run_cfg(tmp_path, output_path=None):
    """small CFG dumped to YAML, optionally with output.path set."""
    cfg = {**CFG}
    if output_path is not None:
        cfg = {**cfg, "output": {"path": str(output_path)}}
    p = tmp_path / "run.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def campaign_calls(monkeypatch):
    from pyofdr.core import campaign

    run_campaign = campaign.run_campaign
    calls = []

    def record_campaign(cfg, keep_acquisitions=True):
        results = run_campaign(cfg, keep_acquisitions=keep_acquisitions)
        # Run the real pipeline, but don't keep its arrays in this fixture.
        calls.append((keep_acquisitions, len(results)))
        return results

    monkeypatch.setattr(campaign, "run_campaign", record_campaign)
    return calls


def test_run_writes_hdf5_when_output_set(tmp_path, campaign_calls):
    out = tmp_path / "out.h5"
    cfg_path = _write_run_cfg(tmp_path, output_path=out)
    code = main(["run", str(cfg_path)])
    assert code == 0
    assert out.exists()
    assert campaign_calls == [(False, 0)]


def test_run_no_output_path_still_succeeds(tmp_path, campaign_calls):
    # no output.path -> nothing written, but exit 0 is fine
    cfg_path = _write_run_cfg(tmp_path)
    code = main(["run", str(cfg_path)])
    assert code == 0
    assert campaign_calls == [(True, 1)]


def test_run_missing_config_exits_2(capsys):
    code = main(["run", "configs/does_not_exist.yaml"])
    assert code == 2


def test_run_output_flag_overrides_config(tmp_path, campaign_calls):
    # config has no output.path, -o picks one
    cfg_path = _write_run_cfg(tmp_path)
    out = tmp_path / "via_flag.h5"
    code = main(["run", str(cfg_path), "-o", str(out)])
    assert code == 0
    assert out.exists()
    assert campaign_calls == [(False, 0)]


def test_run_output_flag_wins_over_config(tmp_path, campaign_calls):
    # config sets a path, -o should win
    in_cfg = tmp_path / "from_cfg.h5"
    cfg_path = _write_run_cfg(tmp_path, output_path=in_cfg)
    out = tmp_path / "from_flag.h5"
    code = main(["run", str(cfg_path), "-o", str(out)])
    assert code == 0
    assert out.exists()
    assert not in_cfg.exists()
    assert campaign_calls == [(False, 0)]


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
