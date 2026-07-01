"""
This module implements a CLI for MatPES, a material property exploration suite.

The CLI provides the following features:
- Download functionality to fetch data related to specific functionals using the `download` command.
- Data processing capabilities using the `data` command, including filtering MatPES data by
  chemical systems or formulas.

The commands are structured as subcommands, facilitating distinct functionalities for
data retrieval and post-processing operations.
"""

from __future__ import annotations

import argparse
import json

from monty.io import zopen
from pymatgen.core import Composition

from .data import get_data


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
    get_data(functional=args.functional, download_atoms=True, return_data=False)
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
                    Defaults to 'filtered.jsonl'.
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
        - The input and output are JSONL files, i.e., one JSON record per line.
          ``zopen`` transparently handles gzip-compressed variants by extension.
    """
    infname = args.filename[0]
    outfname = args.outfile
    with zopen(infname, "rt", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Total number of entries: {len(data)}.")
    if args.chemsys:
        for c in args.chemsys:
            chemsys = "-".join(sorted(c.split("-")))
            data = [d for d in data if d["chemsys"] == chemsys]
    if args.formula:
        for f in args.formula:
            f = Composition(f).reduced_formula
            data = [d for d in data if d["formula_pretty"] == f]
    with zopen(outfname, "wt", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d))
            f.write("\n")
    print(f"{len(data)} filtered entries written in {outfname}.")


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
        default="filtered.jsonl",
        help="File to write filtered entries to (JSONL; use a .gz suffix to gzip-compress).",
    )

    subparser_data.add_argument(
        "filename",
        metavar="filename",
        type=str,
        nargs=1,
        help="Filename to process.",
    )

    subparser_data.set_defaults(func=get_data_subset)

    args = parser.parse_args()

    try:
        _ = args.func
    except AttributeError as exc:
        parser.print_help()
        raise SystemExit("Please specify a command.") from exc
    return args.func(args)
