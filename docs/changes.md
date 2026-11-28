# Change Log

## v0.0.4

### Data

- MatPES is hosted on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes), with 2025.2 (Bader and DDEC6
  charges, duplicates removed) and 2025.3 (8,520 PBE and 7,514 r2SCAN duplicate records of rare earth compounds in the
  wrong low-spin state removed) releases, and the isolated-atom reference energies.
- From 2025.3, MatPES is distributed as Parquet with flat, Arrow-native, float64 columns: about 4x smaller than JSONL,
  more than 10x faster to load, and readable by column or with row filters.
- New repository layout: `main` holds only the latest release as `pbe/` and `r2scan/` (each
  `{train,validation,test}.parquet`, a random 90/5/5 split whose union is the full dataset) and
  `atoms/{pbe,r2scan}.parquet`. Releases are git tags (`v2025.1`, `v2025.2`, `v2025.3`); 2025.1 and 2025.2 remain
  available as JSONL at their tags. The `.json` files and the 2025.3 JSONL files have been removed.
- The dataset card exposes the configs `pbe` (default), `r2scan`, `pbe-atoms` and `r2scan-atoms`, so the Hugging Face
  dataset viewer and `datasets.load_dataset` work, and adds a BibTeX citation.

### Package

- New `matpes.parquet` module: schema, streaming JSONL-to-Parquet conversion, and readers with column projection and
  filters that accept a Parquet file, a folder of Parquet files, a JSONL file or an Arrow table. Files are written with
  content-defined chunking, so future releases deduplicate against earlier ones on Hugging Face.
- New `matpes.convert` module: Parquet/JSONL to pymatgen `Structure`, ASE `Atoms`, MatGL `MGLDataset` or on-disk
  shards, and LMDB (`to_lmdb`, `MatPESLMDB`).
- `get_data()` defaults to the latest release (2025.3), downloads Parquet (2025.1 and 2025.2 quietly fall back to JSONL)
  and still returns a list of record dicts with `d["structure"]`, rebuilt from the Parquet columns. It gains a `split`
  argument (`"train"`, `"valid"`/`"validation"`, `"test"`).
- New in `matpes.data`: `download()` (returns a file, or the folder of split files for the full dataset),
  `get_repo_files()`, `get_revision()`, `load_records()` and `get_atom_energies()`.

### CLI

- New `matpes convert` to convert MatPES-style JSONL files to Parquet.
- `matpes download` fetches Parquet unless `--format jsonl` is given, takes `--version` and `--split`, and prints where
  the files are.
- `matpes data` reads and writes Parquet or JSONL by suffix (Parquet files or folders as input) and defaults to
  `filtered.parquet`. Multiple `--chemsys`/`--formula` values are now OR-ed (previously they were AND-ed, which
  always returned no entries).

### Dependencies

- Python >= 3.12.
- `pyarrow>=21` and `numpy` are explicit dependencies. New optional extras: `ase`, `lmdb`, `matgl`.

## v0.0.3
- Added option to download `atoms` file with atomic reference energies needed for PES fitting.

## v0.0.2
- Update download links.