from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from operational_records import download_month, read_bts_zip, reconstruct_turnarounds


def log(message: str) -> None:
    print(message, flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--airports", nargs="+", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--min-turn", type=int, default=35)
    parser.add_argument("--max-turn", type=int, default=720)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "ctrg" / "raw_bts"
    processed_dir = root / "data" / "ctrg" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    airports = set(args.airports)
    frames = []
    for year in args.years:
        for month in args.months:
            zip_path = download_month(year, month, raw_dir)
            frame = read_bts_zip(zip_path, airports)
            frames.append(frame)
            log(f"Loaded {year}-{month:02d}: {len(frame):,} filtered rows")

    flights = pd.concat(frames, ignore_index=True)
    log(f"Loaded flight rows after airport/tail filter: {len(flights):,}")
    turn = reconstruct_turnarounds(flights, airports, args.min_turn, args.max_turn)
    if turn.empty:
        raise RuntimeError("No turnarounds reconstructed.")

    output_path = processed_dir / args.output_name
    turn.to_csv(output_path, index=False)
    log(f"Reconstructed turnarounds: {len(turn):,}")
    log(f"Saved {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
