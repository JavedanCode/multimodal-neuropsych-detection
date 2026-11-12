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
    run.add_argument(
        "--preprocess-audio",
        action="store_true",
        help="Run the matched waveform cleanup comparison (audio-visual only)",
    )
    run.add_argument(
        "--sparse-fusion",
        action="store_true",
        help="Run the final sparse/weighted/reliability comparison (audio-visual only)",
    )
    run.add_argument(
        "--corrected",
        action="store_true",
        help="Run the voiced-audio and common-scale fusion correction (audio-visual only)",
    )
    run.add_argument(
        "--score-study",
        action="store_true",
        help="Run the PHQ-score / eGeMAPS functionals / domain scaling study (audio-visual only)",
    )
    run.add_argument(
        "--pooled-study",
        action="store_true",
        help="Run the participant-speech / pooled train+dev / text-assisted study (audio-visual only)",
    )
    run.add_argument(
        "--stacked-study",
        action="store_true",
        help="Run the stacked text+voice+face fusion under three fold schemes (audio-visual only)",
    )
    verify = commands.add_parser("verify", help="Check a saved study's artifacts")
    verify.add_argument("study", choices=STUDIES)
    verify.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    module_name = STUDIES[args.study]
    if args.command == "run" and args.preprocess_audio and args.study != "audio-visual":
        parser.error("--preprocess-audio is available only for audio-visual")
    if args.command == "run" and args.sparse_fusion:
        if args.study != "audio-visual" or args.preprocess_audio:
            parser.error(
                "--sparse-fusion requires audio-visual and cannot combine with --preprocess-audio"
            )
    if args.command == "run" and args.stacked_study:
        if args.study != "audio-visual" or (
            args.preprocess_audio
            or args.sparse_fusion
            or args.corrected
            or args.score_study
            or args.pooled_study
        ):
            parser.error("--stacked-study requires audio-visual and runs on its own")
    if args.command == "run" and args.pooled_study:
        if args.study != "audio-visual" or (
            args.preprocess_audio or args.sparse_fusion or args.corrected or args.score_study
        ):
            parser.error("--pooled-study requires audio-visual and runs on its own")
    if args.command == "run" and args.score_study:
        if args.study != "audio-visual" or (
            args.preprocess_audio or args.sparse_fusion or args.corrected
        ):
            parser.error("--score-study requires audio-visual and runs on its own")
    if args.command == "run" and args.corrected:
        if args.study != "audio-visual" or args.preprocess_audio or args.sparse_fusion:
            parser.error("--corrected requires audio-visual and runs on its own")
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
        elif args.preprocess_audio:
            print(module.run(project, preprocessing=True))
        elif args.stacked_study:
            print(module.run(project, stacked_study=True))
        elif args.pooled_study:
            print(module.run(project, pooled_study=True))
        elif args.score_study:
            print(module.run(project, score_study=True))
        elif args.corrected:
            print(module.run(project, corrected=True))
        elif args.sparse_fusion:
            print(module.run(project, sparse_fusion=True))
        elif args.study in {"text-candidates", "fallback"}:
            print(module.run(project, args.reference_run.resolve()))
        else:
            print(module.run(project))


if __name__ == "__main__":
    main()
