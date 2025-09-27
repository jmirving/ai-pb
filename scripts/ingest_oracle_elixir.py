"""CLI helper for materializing Oracle's Elixir data locally.

The training pipeline expects cleaned Oracle's Elixir data to be cached in
``data/processed``.  This script delegates to :class:`pbai.data` services to
read the CSV export, store a versioned copy, and create convenience slices
for teams and players.  Running it regularly keeps the cache synchronized
with the raw CSV that ships in ``resources`` (or a custom path supplied via
``--source-path``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime


# Ensure project root is importable when the script is invoked directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pbai.data.data_ingestion_service import DataIngestionService
from pbai.data.storage import FileRepository


def ingest_oracle_elixir(
    source_path: str,
    processed_dir: str = "data/processed",
    force_refresh: bool = False,
) -> dict:
    """Materialize Oracle's Elixir data and return a short summary.

    Args:
        source_path: Path to the Oracle's Elixir CSV export.
        processed_dir: Directory where parquet caches should live.
        force_refresh: Whether to ignore cached parquet files and reload
            the CSV even if it has not changed.

    Returns:
        Dictionary containing counts for the datasets that were cached.
    """

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"Oracle's Elixir CSV not found at '{source_path}'. "
            "Pass --source-path to point to the file."
        )

    logging.info("Loading Oracle's Elixir export from %s", source_path)
    repository = FileRepository(processed_dir)
    service = DataIngestionService(repository, source_path)

    # ``get_all_data_df`` performs the heavy lifting of saving a versioned copy
    # to disk whenever the CSV is newer than the cached parquet.
    all_df = service.get_all_data_df(force_refresh=force_refresh)
    logging.info("Loaded %d rows of raw match data", len(all_df))

    # Player and team slices are cached explicitly here so downstream stages can
    # reuse them without repeating the filtering work.
    players_df = service.get_players_df()
    teams_df = service.get_teams_df()

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    repository.save_raw_player_data(players_df, version)
    repository.save_raw_team_data(teams_df, version)
    repository.cleanup_old_versions()

    summary = {
        "all_rows": len(all_df),
        "player_rows": len(players_df),
        "team_rows": len(teams_df),
        "processed_dir": processed_dir,
    }
    logging.info(
        "Oracle's Elixir data cached (all=%(all_rows)d, players=%(player_rows)d, "
        "teams=%(team_rows)d) in %(processed_dir)s",
        summary,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Oracle's Elixir match data for downstream stages."
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
        "--force-refresh",
        action="store_true",
        help="Ignore cached parquet files and reload the CSV from disk.",
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

    summary = ingest_oracle_elixir(
        source_path=args.source_path,
        processed_dir=args.processed_dir,
        force_refresh=args.force_refresh,
    )

    print("Oracle's Elixir data cached:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
