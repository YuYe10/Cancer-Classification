#!/usr/bin/env python3
"""Convert TSV/TSV.GZ file(s) to CSV.

Supports:
- Single file conversion (.tsv or .tsv.gz)
- Batch conversion for all .tsv.gz files in a directory
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import TextIO, cast


def open_text(path: Path, mode: str):
    """Open plain text or gzipped text file with UTF-8 encoding."""
    if path.suffix == ".gz":
        return gzip.open(path, mode=mode, encoding="utf-8", newline="")
    return path.open(mode=mode, encoding="utf-8", newline="")


def tsv_to_csv(input_path: Path, output_path: Path) -> None:
    with open_text(input_path, "rt") as fin, open_text(output_path, "wt") as fout:
        text_in = cast(TextIO, fin)
        text_out = cast(TextIO, fout)
        reader = csv.reader(text_in, delimiter="\t")
        writer = csv.writer(text_out)
        for row in reader:
            writer.writerow(row)


def batch_convert_dir(input_dir: Path, recursive: bool = False) -> int:
    pattern = "**/*.tsv.gz" if recursive else "*.tsv.gz"
    input_files = sorted(input_dir.glob(pattern))

    for input_file in input_files:
        output_file = default_output_path(input_file)
        tsv_to_csv(input_file, output_file)
        print(f"Converted: {input_file} -> {output_file}")

    return len(input_files)


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".tsv.gz"):
        return input_path.with_name(name[:-7] + ".csv")
    if name.endswith(".tsv"):
        return input_path.with_name(name[:-4] + ".csv")
    return input_path.with_suffix(".csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one .tsv/.tsv.gz file or batch convert a directory"
    )
    parser.add_argument("input", type=Path, help="Input file or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output .csv file")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search .tsv.gz files when input is a directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input

    if input_path.is_dir():
        if args.output is not None:
            raise SystemExit("--output is only valid for single file conversion.")
        count = batch_convert_dir(input_path, recursive=args.recursive)
        print(f"Done. Converted {count} file(s).")
        return

    if not input_path.exists():
        raise SystemExit(f"Input does not exist: {input_path}")

    output_path: Path = args.output or default_output_path(input_path)
    tsv_to_csv(input_path, output_path)
    print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
