"""
This module implements a CLI for MatPES, a material property exploration suite.

The CLI provides the following features:
- Download functionality to fetch data related to specific functionals using the `download` command.
- Data processing capabilities using the `data` command, including filtering MatPES data by
  chemical systems or formulas. Both JSONL and Parquet files are supported.
- Conversion of JSONL files to the flat Parquet format using the `convert` command.

The commands are structured as subcommands, facilitating distinct functionalities for
data retrieval and post-processing operations.
"""

from __future__ import annotations

import argparse
import json

import pyarrow.compute as pc
from monty.io import zopen
from pymatgen.core import Composition

from .data import get_data
from .parquet import is_parquet, jsonl_to_parquet, read_table, record_to_row, write_parquet


def download(args):
    """
    Function to download data based on the given functional argument.

    This function utilizes the "get_data" method with an input argument
    to fetch and download data tied to the specified functionality. After
    successful execution, it outputs a confirmation message.

    Parameters
    ----------
    args : argparse.Namespace
        Argument namespace that must contain a "functional" attribute.

    Raises:
    ------
    None
    """
    get_data(functional=args.functional, download_atoms=True, return_data=False, fmt=args.format)
    print(f">>> Successfully downloaded data for {args.functional}.")


def get_data_subset(args):
    """
    Filter MatPES data by chemical system or formula.

    This function processes a given JSONL file containing MatPES data and filters
    the entries based on chemical systems or formulas specified by the user.
    The filtered results are then written to an output file.

    Parameters:
        args: Namespace
            A namespace object containing the following attributes:
                - filename: List[str]
                    List containing the name of the input file to process (only
                    the first entry is used).
                - outfile: str
                    Name of the output file to write the filtered results to.
                    Defaults to 'filtered.parquet'.
                - chemsys: List[str]
                    List of chemical systems (e.g., 'Li-Fe-O') to filter by.
                    If empty, no filtering by chemical system is applied.
                - formula: List[str]
                    List of formulas (e.g., 'Fe2O3') to filter by. If empty,
                    no filtering by formula is applied.

    Returns:
        None

    Notes:
        - The chemical system string should follow the format 'Element1-Element2-...'.
        - Formulas are case-insensitive and automatically converted to their
          reduced forms for comparison.
        - Multiple chemical systems or formulas are combined with OR; chemsys and formula filters with AND.
        - Input and output can be JSONL (``zopen`` handles gzip by extension) or Parquet, chosen by suffix.
          JSONL input can be written as Parquet, but not vice versa.
    """
    infname = args.filename[0]
    outfname = args.outfile
    chemsys = ["-".join(sorted(c.split("-"))) for c in args.chemsys or []]
    formulas = [Composition(f).reduced_formula for f in args.formula or []]

    if is_parquet(infname):
        if not is_parquet(outfname):
            raise SystemExit("Parquet input requires a Parquet output file (e.g., -o filtered.parquet).")
        expr = None
        if chemsys:
            expr = pc.field("chemsys").isin(chemsys)
        if formulas:
            f_expr = pc.field("formula_pretty").isin(formulas)
            expr = f_expr if expr is None else expr & f_expr
        table = read_table(infname, filters=expr)
        n = write_parquet(
            table, outfname, metadata={"source": infname, "filter": {"chemsys": chemsys, "formula": formulas}}
        )
        print(f"{n} filtered entries written in {outfname}.")
        return

    with zopen(infname, "rt", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Total number of entries: {len(data)}.")
    if chemsys:
        data = [d for d in data if d["chemsys"] in chemsys]
    if formulas:
        data = [d for d in data if d["formula_pretty"] in formulas]
    if is_parquet(outfname):
        write_parquet((record_to_row(d) for d in data), outfname, metadata={"source": infname})
    else:
        with zopen(outfname, "wt", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d))
                f.write("\n")
    print(f"{len(data)} filtered entries written in {outfname}.")


def convert(args):
    """Convert MatPES JSONL files into the flat Parquet format (see :mod:`matpes.parquet`)."""
    for fname in args.filenames:
        metadata = {"version": args.version} if args.version else None
        out = jsonl_to_parquet(fname, args.outfile if len(args.filenames) == 1 else None, metadata=metadata)
        print(f">>> {fname} -> {out}")


def main():
    """Main entry point for matpes cli."""
    parser = argparse.ArgumentParser(
        description="""matpes is a CLI for MatPES.""",
        epilog="Author: Shyue Ping Ong",
    )

    subparsers = parser.add_subparsers()
    subparser_download = subparsers.add_parser(name="download")
    subparser_download.add_argument(
        "functional",
        metavar="functional",
        type=str.upper,
        nargs="?",
        default="PBE",
        help="Functional to download. Defaults to PBE.",
    )

    subparser_download.add_argument(
        "--format",
        dest="format",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="File format. Defaults to parquet (JSONL for versions without Parquet files).",
    )

    subparser_download.set_defaults(func=download)

    subparser_data = subparsers.add_parser(
        name="data", help="Process downloaded MatPES data files, e.g., filtering by chemical system or formula."
    )

    subparser_data.add_argument(
        "-f",
        "--formula",
        dest="formula",
        type=str,
        nargs="*",
        help="Formulas to filter by, e.g., Fe2O3.",
    )

    subparser_data.add_argument(
        "-c",
        "--chemsys",
        dest="chemsys",
        type=str,
        nargs="*",
        help="Chemical systems to filter by, e.g., Li-Fe-O.",
    )

    subparser_data.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        type=str,
        nargs="?",
        default="filtered.parquet",
        help="File to write filtered entries to. Format by suffix: .parquet (default) or .jsonl (.gz to compress).",
    )

    subparser_data.add_argument(
        "filename",
        metavar="filename",
        type=str,
        nargs=1,
        help="Filename to process.",
    )

    subparser_data.set_defaults(func=get_data_subset)

    subparser_convert = subparsers.add_parser(name="convert", help="Convert MatPES JSONL files to Parquet.")
    subparser_convert.add_argument("filenames", metavar="filenames", type=str, nargs="+", help="JSONL files.")
    subparser_convert.add_argument(
        "-o", "--outfile", dest="outfile", type=str, default=None, help="Output file (single input only)."
    )
    subparser_convert.add_argument(
        "-v", "--version", dest="version", type=str, default=None, help="Dataset version to record in metadata."
    )
    subparser_convert.set_defaults(func=convert)

    args = parser.parse_args()

    try:
        _ = args.func
    except AttributeError as exc:
        parser.print_help()
        raise SystemExit("Please specify a command.") from exc
    return args.func(args)
