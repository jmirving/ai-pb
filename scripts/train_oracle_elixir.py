"""CLI entry point for training the Oracle's Elixir draft model."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pbai.training.train_oracle_elixir import train_oracle_elixir


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Oracle's Elixir draft model using cached data.",
    )
    parser.add_argument(
        "--oracle-elixir-path",
        default=os.path.join(
            PROJECT_ROOT,
            "resources",
            "2025_LoL_esports_match_data_from_OraclesElixir.csv",
        ),
        help="Path to the Oracle's Elixir CSV export.",
    )
    parser.add_argument(
        "--processed-data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "processed"),
        help="Directory containing cached parquet data.",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(
            PROJECT_ROOT,
            "models",
            "draft_mlp_oracle_elixir.pth",
        ),
        help="Destination path for the trained model checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for the data loader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--ingestion-export-format",
        dest="ingestion_export_formats",
        action="append",
        choices=["csv"],
        help=(
            "Additional formats (besides parquet) to materialize when refreshing cached data."
        ),
    )
    parser.add_argument(
        "--dataset-export-dir",
        help="Optional directory where DraftDataset will export intermediate CSV snapshots.",
    )
    parser.add_argument(
        "--dataset-export-format",
        dest="dataset_export_formats",
        action="append",
        choices=["csv"],
        help="Additional formats for DraftDataset intermediate exports.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for log messages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    train_oracle_elixir(
        oracle_elixir_path=args.oracle_elixir_path,
        processed_data_dir=args.processed_data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        ingestion_export_formats=args.ingestion_export_formats,
        dataset_export_dir=args.dataset_export_dir,
        dataset_export_formats=args.dataset_export_formats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
