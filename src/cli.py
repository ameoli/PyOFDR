"""Command-line interface (issue #29).

Run as::

    PYTHONPATH=src python -m cli info     configs/ofdr_basic.yaml
    PYTHONPATH=src python -m cli validate configs/ofdr_basic.yaml
    PYTHONPATH=src python -m cli run      configs/ofdr_basic.yaml

Once the package layout is reshuffled for PyPI (#55) this will move
under pyofdr.cli and become a proper console entry point.
"""

from __future__ import annotations

import argparse
import sys

from core.config import load_config, print_info


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
        # campaign import is local: keeps `pyofdr info` / `validate` snappy
        # by skipping the heavy pipeline imports
        from core.campaign import run_campaign
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
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
