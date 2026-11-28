# Change Log

## v0.0.4

- New dataset release on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes): removed 8,520 PBE and
  7,514 r2SCAN duplicate records of rare earth compounds that are in the wrong low-spin state.
- `get_data()` now defaults to `version="2025.3"`.
- Website download buttons, versions table and docs updated to 2025.3.
- MatPES 2025.3 is now distributed on Hugging Face in Parquet format with flat, Arrow-native, float64 columns.
- The Hugging Face repository is reorganized: `main` holds only the latest release as `pbe/`, `r2scan/` (each
  `train`/`validation`/`test`.parquet; the full dataset is their union) and `atoms/{pbe,r2scan}.parquet`. Releases are
  git tags (`v2025.1`, `v2025.2`, `v2025.3`); 2025.1 and 2025.2 remain available as JSONL at their tags. The `.json`
  files and the 2025.3 JSONL files have been removed.
- New `matpes.parquet` module: schema, streaming JSONL-to-Parquet conversion, and readers with column projection and
  filters. All readers accept JSONL as well.
- New `matpes.convert` module: Parquet/JSONL to pymatgen `Structure`, ASE `Atoms`, MatGL `MGLDataset` or on-disk
  shards, and LMDB (`to_lmdb`, `MatPESLMDB`).
- `get_data()` now downloads Parquet by default (2025.1 and 2025.2 quietly fall back to JSONL) and still returns a
  list of record dicts with `d["structure"]`, rebuilt from the Parquet columns. New `download()` (returns a file, or
  the folder of split files for the full dataset), `get_repo_files()`, `get_revision()` and `get_atom_energies()` in
  `matpes.data`; `get_data()` gains a `split` argument. All readers accept a folder of Parquet files.
- Parquet files are written with content-defined chunking, so future releases deduplicate against earlier ones on
  Hugging Face.
- CLI: new `matpes convert`; `matpes download` fetches Parquet unless `--format jsonl` is given, takes `--version`
  and `--split`, and prints where the files are; `matpes data` reads
  and writes Parquet or JSONL by suffix and defaults to `filtered.parquet`. Multiple
  `--chemsys`/`--formula` values are now OR-ed (previously they were AND-ed, which always returned no entries).
- New optional extras: `ase`, `lmdb`, `matgl`. `pyarrow>=21` is now an explicit dependency.

## v0.0.3
- Added option to download `atoms` file with atomic reference energies needed for PES fitting.

## v0.0.2
- Update download links.