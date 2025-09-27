"""Run the full data-to-model pipeline in a single command."""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root and scripts directory are importable when running directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for path in {PROJECT_ROOT, SCRIPTS_DIR}:
    if path not in sys.path:
        sys.path.insert(0, path)

from ingest_oracle_elixir import ingest_oracle_elixir
from pbai.training.train_oracle_elixir import train_oracle_elixir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute ingestion and model training back-to-back."
    )
    parser.add_argument(
        "--source-path",
        default=os.path.join(
            PROJECT_ROOT,
            "resources",
            "2025_LoL_esports_match_data_from_OraclesElixir.csv",
        ),
        help="Path to the Oracle's Elixir CSV export.",
    )
    parser.add_argument(
        "--processed-dir",
        default=os.path.join(PROJECT_ROOT, "data", "processed"),
        help="Directory where parquet caches will be written.",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(
            PROJECT_ROOT, "models", "draft_mlp_oracle_elixir.pth"
        ),
        help="Destination path for the trained model weights.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size used during training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Assume data is already cached and skip the ingestion stage.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for log messages.",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.skip_ingestion:
        logging.info("Skipping ingestion stage (per --skip-ingestion)")
    else:
        ingest_summary = ingest_oracle_elixir(
            source_path=args.source_path,
            processed_dir=args.processed_dir,
            force_refresh=False,
        )
        logging.info("Ingestion summary: %s", ingest_summary)

    logging.info("Starting training stage")
    train_oracle_elixir(
        oracle_elixir_path=args.source_path,
        processed_data_dir=args.processed_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
