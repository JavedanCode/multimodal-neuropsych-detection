"""One entry point for the maintained research comparisons."""

import argparse
import importlib
import json
import logging
from pathlib import Path

from threadpoolctl import threadpool_limits

STUDIES = {
    "reference": "reference",
    "text-candidates": "text_candidates",
    "audio-visual": "audio_visual",
    "fallback": "fallback",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Train and evaluate a maintained study")
    run.add_argument("study", choices=STUDIES)
    run.add_argument("--reference-run", type=Path)
    verify = commands.add_parser("verify", help="Check a saved study's artifacts")
    verify.add_argument("study", choices=STUDIES)
    verify.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    module_name = STUDIES[args.study]
    if args.command == "run" and args.study in {"text-candidates", "fallback"}:
        if args.reference_run is None:
            parser.error(f"{args.study} requires --reference-run")
    package = "evaluation.verify_" if args.command == "verify" else "experiments."
    module = importlib.import_module(f"src.{package}{module_name}")
    # Load numerical libraries before setting their thread limits, so newly
    # imported BLAS runtimes also receive the reproducible single-thread setting.
    with threadpool_limits(limits=1):
        if args.command == "verify":
            print(json.dumps(module.verify(project, args.run.resolve()), indent=2))
        elif args.study in {"text-candidates", "fallback"}:
            print(module.run(project, args.reference_run.resolve()))
        else:
            print(module.run(project))


if __name__ == "__main__":
    main()
