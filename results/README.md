# results/

- `raw/` — raw JSON/CSV emitted by scripts. Committed deliberately: these are the
  primary record. Every row carries model revision, seed, corpus draw, git commit and
  timestamp.
- `tables/` — generated summary tables. Regenerable; never hand-edited.

Large binary artifacts (`.safetensors`, `.pt`) are gitignored.
