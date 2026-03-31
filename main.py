from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from quantum_coughfreq.cli import (
    classify_dataset_command,
    dataset_summary_command,
    predict_command,
    serve_command,
    tabular_summary_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantum CoughFreq terminal runner")
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser("predict", help="Run cough prediction on a .wav file")
    predict_parser.add_argument("audio_file", help="Path to the input audio file")
    predict_parser.add_argument("--json", action="store_true", dest="json_output", help="Print raw JSON output")

    classify_parser = subparsers.add_parser(
        "classify-dataset",
        help="Recursively classify all supported audio files in a dataset folder",
    )
    classify_parser.add_argument("dataset_dir", help="Dataset folder containing cough audio files")
    classify_parser.add_argument("--json", action="store_true", dest="json_output", help="Print raw JSON output")
    classify_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-file predictions in terminal output",
    )
    classify_parser.add_argument(
        "--top-k",
        default=10,
        type=int,
        help="How many highest TB-risk files to show in the summary",
    )
    classify_parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="How many parallel workers to use while classifying the dataset",
    )
    classify_parser.add_argument(
        "--max-files",
        default=None,
        type=int,
        help="Optional limit for faster trial runs on very large datasets",
    )

    dataset_parser = subparsers.add_parser(
        "dataset-summary",
        help="Inspect dataset folder structure before training",
    )
    dataset_parser.add_argument("dataset_dir", help="Dataset root directory")

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server only when needed")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", default=8011, type=int, help="Bind port")

    tabular_parser = subparsers.add_parser(
        "tabular-summary", help="Analyze pre-extracted tabular features from CSV"
    )
    tabular_parser.add_argument("csv_file", help="Path to the features .csv file")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict":
        return predict_command(args.audio_file, json_output=args.json_output)
    if args.command == "classify-dataset":
        return classify_dataset_command(
            args.dataset_dir,
            json_output=args.json_output,
            detailed=args.detailed,
            top_k=args.top_k,
            workers=args.workers,
            max_files=args.max_files,
        )
    if args.command == "dataset-summary":
        return dataset_summary_command(args.dataset_dir)
    if args.command == "serve":
        return serve_command(host=args.host, port=args.port)
    if args.command == "tabular-summary":
        return tabular_summary_command(args.csv_file)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
