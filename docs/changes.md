# Change Log

## v0.0.4

- New dataset release on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes): removed 8,520 PBE and
  7,514 r2SCAN duplicate records of rare earth compounds that are in the wrong low-spin state.
- `get_data()` now defaults to `version="2025.3"`.
- Website download buttons, versions table and docs updated to 2025.3.
- MatPES 2025.3 is now distributed on Hugging Face in Parquet format with flat, Arrow-native, float64 columns
  (`MatPES-<PBE|R2SCAN>-2025.3[-train|-valid|-test].parquet`, `MatPES-<PBE|R2SCAN>-atoms.parquet`). JSONL files remain
  available for all versions; the redundant `.json` files of 2025.1 and 2025.2 have been removed.
- New `matpes.parquet` module: schema, streaming JSONL-to-Parquet conversion, and readers with column projection and
  filters. All readers accept JSONL as well.
- New `matpes.convert` module: Parquet/JSONL to pymatgen `Structure`, ASE `Atoms`, MatGL `MGLDataset` or on-disk
  shards, and LMDB (`to_lmdb`, `MatPESLMDB`).
- `get_data()` now downloads Parquet by default (versions without Parquet quietly fall back to JSONL; `fmt="jsonl"`
  forces JSONL) and still returns a list of record dicts with `d["structure"]`, rebuilt from the Parquet columns. New `download()`,
  `get_filename()` and `get_atom_energies()` in `matpes.data`; `get_data()` gains a `split` argument.
- CLI: new `matpes convert`; `matpes download` fetches Parquet unless `--format jsonl` is given; `matpes data` reads
  and writes Parquet or JSONL by suffix and defaults to `filtered.parquet`. Multiple
  `--chemsys`/`--formula` values are now OR-ed (previously they were AND-ed, which always returned no entries).
- New optional extras: `ase`, `lmdb`, `matgl`. `pyarrow` is now an explicit dependency.

## v0.0.3
- Added option to download `atoms` file with atomic reference energies needed for PES fitting.

## v0.0.2
- Update download links.