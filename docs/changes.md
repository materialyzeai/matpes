# Change Log

## v0.0.4

- New dataset release on [Hugging Face](https://huggingface.co/datasets/materialyze/matpes): removed 8,520 PBE and
  7,514 r2SCAN records that duplicate the structure of another record; added a random 90/5/5 train/valid/test split
  (`MatPES-{PBE,R2SCAN}-2025.3-{train,valid,test}.jsonl`).
- `get_data()` now defaults to `version="2025.3"`.
- Website download buttons, versions table and docs updated to 2025.3.

## v0.0.3
- Added option to download `atoms` file with atomic reference energies needed for PES fitting.

## v0.0.2
- Update download links.