"""Command-line interface (issue #29).

For now only `info` is wired up. `validate` and `run` will follow.

Run as::

    PYTHONPATH=src python -m cli info configs/ofdr_basic.yaml

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

    # argparse should have rejected us before getting here
    return 1


if __name__ == "__main__":
    sys.exit(main())
