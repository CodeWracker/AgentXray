# REPRODUCIBILITY AND ISOLATION

Your run directory is `{{RUN_DIR}}`. Work only inside it. Do not read, list or search files outside it (other than the `torchxrayvision` source code, the model directory `{{MODELS_DIR}}` and the Python environment). Other directories of this repository contain results of previous runs, and reading them would contaminate your independent analysis.

Every analysis folder must contain `scripts/reproduce.py`, a single entrypoint that:

* takes no arguments and is run from the analysis folder, as `{{PYTHON}} scripts/reproduce.py`
* reads only `images/00_original.*` (and optionally other scripts in `scripts/`)
* regenerates every file in `images/` (except `00_original.*`) and every file in `measurements/`
* is deterministic: fix every random seed (`random`, `numpy`, `torch`), call `torch.use_deterministic_algorithms(True)`, and do not write timestamps, absolute paths or other run-dependent values into its outputs

Your exploration may use any number of scripts in any order, but anything you keep in `images/` or `measurements/` must be reproducible by `scripts/reproduce.py`. Delete outputs you no longer want before the final validation, and record in `provenance/decisions.md` (or `planning/processing_plan.txt`) that you discarded them and why.

Before finishing, delete every file in `images/` (except `00_original.*`) and in `measurements/`, run `scripts/reproduce.py`, and confirm that all outputs were recreated. An automated checker will later run it again in a clean copy and compare the files byte by byte.

`provenance/tools.json` must list, as a JSON array, every library function and pretrained model whose output you relied on, each as an object with `tool`, `used_for` and `trusted` (`true` or `false`, with `reason`). Tools you tried and rejected also belong there, with `trusted: false`.
