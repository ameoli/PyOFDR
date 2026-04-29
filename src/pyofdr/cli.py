"""Command-line interface (issue #29).

Run as::

    pyofdr info     configs/ofdr_basic.yaml
    pyofdr validate configs/ofdr_basic.yaml
    pyofdr run      configs/ofdr_basic.yaml
"""

from __future__ import annotations

import argparse
import sys

from pyofdr.core.config import load_config, print_info


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyofdr",
                                 description="OFDR simulator CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="print a summary of a config file")
    p_info.add_argument("config", help="path to YAML config")

    p_val = sub.add_parser("validate", help="load + validate a config, exit 0 if ok")
    p_val.add_argument("config", help="path to YAML config")

    p_run = sub.add_parser("run", help="run a simulation from a config")
    p_run.add_argument("config", help="path to YAML config")
    p_run.add_argument("-o", "--output", default=None,
                       help="HDF5 output path (overrides output.path in the config)")
    p_run.add_argument("-q", "--quiet", action="store_true",
                       help="suppress progress logs (only warnings/errors)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "info":
        try:
            cfg = load_config(args.config)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print_info(cfg)
        return 0

    if args.cmd == "validate":
        try:
            load_config(args.config)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            # pydantic ValidationError, yaml errors, etc. don't dump the
            # traceback on the user -- they just want to know it's broken.
            print(f"invalid config: {e}", file=sys.stderr)
            return 1
        print(f"{args.config}: OK")
        return 0

    if args.cmd == "run":
        try:
            cfg = load_config(args.config)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"invalid config: {e}", file=sys.stderr)
            return 1
        if args.output is not None:
            cfg.setdefault("output", {})["path"] = args.output
        # campaign import is local: keeps `pyofdr info` / `validate` snappy
        # by skipping the heavy pipeline imports
        from pyofdr.core.campaign import run_campaign
        import logging
        level = logging.WARNING if args.quiet else logging.INFO
        # basicConfig is a no-op once handlers exist (e.g. from a prior
        # call), so set the root level explicitly. format only matters
        # the first time around.
        logging.basicConfig(level=level, format="%(message)s")
        logging.getLogger().setLevel(level)
        try:
            run_campaign(cfg)
        except Exception as e:
            print(f"run failed: {e}", file=sys.stderr)
            return 1
        return 0

    # argparse should have rejected us before getting here
    return 1


if __name__ == "__main__":
    sys.exit(main())
