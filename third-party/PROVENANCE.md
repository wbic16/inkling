# Third-party provenance

Everything in this directory is someone else's work, kept because the run
depended on it byte-for-byte. It is **not** covered by this repo's LICENSE;
each item remains under its upstream terms.

## rse_looper — Eric E. / Tensorbend.ai

- Upstream: https://github.com/eelbaz/Recursive-Self-Improving-Loopers
- Commit: `765453d13878804c49876a656f39b57a4384dca4` (see `rse_looper/COMMIT.txt`)
- Contents: `rse_looper/` (the package as received: `fpsolve.py`,
  `looping.py`, `checkpoint.py`, `gspo.py`, `grader.py`, `pipeline.py`,
  `preflight.py`, `__init__.py`, `pyproject.toml`) and `rse-pkg.tgz`
  (the tarball exactly as received, sha256-verifiable against the unpacked
  copy).
- **Byte-unmodified.** `fpsolve.py` sha256
  `15510672e457b8dd215026cef50d3f0db6b82516e5d2f25704848e438635da26`.
  Our changes live outside this directory and are recorded in the repo's
  `PATCHES.md` (shims S1–S4, ports P1–P3, transport T1).

## llama.cpp-inkling2.py — the llama.cpp project

- The `inkling2` model definition from llama.cpp's gguf conversion
  tooling (imports `gguf`, `.base`), staged during the applicability recon
  for GGUF census/conversion reference. Kept for reference; the fleet
  runtime is llama.cpp itself, not this file.
