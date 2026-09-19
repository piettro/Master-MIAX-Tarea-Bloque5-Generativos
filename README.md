# Synthetic Financial Data for Drawdown Prediction

> Master's in Artificial Intelligence Applied to Financial Markets
> Course: Generative AI | Assignment: B5-T1 -- Synthetic Financial Data Generation

## Overview

Can synthetic data improve a classifier that predicts severe drawdowns of
the S&P 500? Using 60-day windows of daily log returns from a 23-asset
universe, the classifier predicts whether the equal-weighted portfolio will
suffer a drawdown worse than its own 10th-percentile threshold over the
following 30 days.

Three genuinely different generative families -- a class-conditional GAN, a
class-conditional VAE, and a class-conditional denoising-diffusion model --
plus a mandatory simple noise-perturbation baseline are each trained on a
fixed, deliberately scarce budget of 3,000 real windows. Their synthetic
output is then mixed into that budget at increasing ratios and the
classifier is retrained and evaluated at every ratio, with five random
seeds each, against a held-out, chronologically later test set.

**This is a refactor of an already-submitted, already-graded group project**
(see [Provenance](#provenance) below), rebuilt as an installable Python
package with type hints, logging, tests, and a single CLI entry point,
while cross-referencing the professor's own course material and the
recorded correction lecture for issues the original notebooks missed.

## Project Structure

```
B5-T1-refactored/
├── main.py                          CLI entry point (see Usage)
├── requirements.txt                 pinned dependencies
├── src/
│   ├── utils/
│   │   ├── config.py                 all paths, hyperparameters, constants
│   │   └── logging_config.py         root logger setup
│   ├── data/
│   │   ├── loader.py                  price download and caching
│   │   └── dataset.py                 returns, windows, labels, split, scaling
│   ├── models/
│   │   ├── classifier.py              CNN classifier, training, architecture search
│   │   └── generators/
│   │       ├── base.py                 BaseGenerator interface
│   │       ├── noise.py                mandatory simple baseline
│   │       ├── gan.py                  class-conditional GAN
│   │       ├── vae.py                  class-conditional VAE
│   │       └── diffusion.py            class-conditional diffusion model
│   ├── evaluation/
│   │   ├── experiment.py               ratio sweep, mixing, persistence
│   │   ├── statistics.py               paired significance test, comparison table
│   │   └── mutual_information.py       label-information analysis
│   └── visualization/
│       └── plots.py                    every shared figure
├── tests/                            pytest unit tests (40 tests, no network calls)
├── docs/
│   └── architecture.md               Mermaid diagrams, data flow, design notes
├── notebooks/original_submission/    the eight notebooks as originally delivered
├── professor_notebooks/              course reference material (kept as-is)
├── materials/                        assignment brief, class summary, correction transcript
└── data/ · models/ · results/        generated artifacts (see .gitignore)
```

## Setup & Installation

```bash
# 1. Clone or copy this folder
cd B5-T1-refactored

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Python 3.14 has no TensorFlow wheel; Keras 3 runs on the PyTorch backend
instead (`KERAS_BACKEND=torch`), which every entry point sets automatically.
Everything runs on CPU.

**No download is required to reproduce the study.** The repository ships
with the small, already-filtered `data/processed/prices_universe.csv`
(the 23 assets with full history since 1945). The full raw download
(`python main.py download-prices`) is only needed to change the starting
universe.

## Usage

```bash
python main.py build-dataset            # ~1 min  -- builds dataset.npz
python main.py threshold-sensitivity    # seconds -- sweeps the crisis threshold
python main.py search-architecture      # ~10 min -- classifier architecture grid
python main.py train-baseline           # ~1 min  -- reference, no synthetics

python main.py train-generator noise        # seconds
python main.py evaluate-generator noise     # ~5 min

python main.py train-generator cgan         # ~8 min
python main.py evaluate-generator cgan      # ~8 min

python main.py train-generator cvae         # ~4 min
python main.py evaluate-generator cvae      # ~8 min

python main.py train-generator diffusion    # ~20-35 min (the slow one)
python main.py evaluate-generator diffusion # ~15 min

python main.py compare                  # seconds -- final tables and figures
```

or, end to end (the diffusion generator dominates the run time):

```bash
python main.py run-all                  # everything above, in order
python main.py run-all --skip-diffusion # skip the slowest stage
```

Every `train-generator`/`evaluate-generator` pair is independent once
`build-dataset` has run, and can be re-run in any order. `compare` only
needs whatever `results/tables/results_<name>.csv` files already exist.

Run the test suite (fast: no network access, no full-size training):

```bash
pytest tests/ -v
```

## Results

Running the full pipeline produces, under `results/`:

* `figures/` -- loss/convergence curves per generator, real-vs-synthetic
  trajectory and distribution comparisons, the central ratio-vs-PR-AUC
  figure (test and validation), and the mutual-information breakdown.
* `tables/` -- one `results_<name>.csv` per generator (one row per ratio and
  seed), `architecture_search.csv`, `threshold_sensitivity.csv`,
  `contrasts.csv` (paired significance test), `final_comparison.csv`, and
  `mutual_information.csv`.

The headline finding from the original run (see
[Provenance](#provenance)): **no generator improved the classifier by a
statistically significant margin**, and the amount of label-relevant
information each generator's synthetic samples carry (measured via mutual
information against six interpretable window descriptors) tracks closely
with how little each one helped. The full narrative, with numbers, lives in
the original project's `README.md` and presentation, referenced below.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the module dependency
graph, the data-flow diagram, the generator class hierarchy, and detailed
design notes (in particular, why the diffusion generator predicts the clean
window instead of the added noise -- a correctness fix, not a style choice).

## Provenance

This package is a from-scratch, English-language, engineering-focused
rewrite of a project that was already designed, implemented, delivered, and
orally defended as a Jupyter-notebook-based group submission (Raúl, Pietro,
Alonso) by the assignment's 3 September deadline. The original submission --
Spanish, notebook-first, matching every explicit instruction in
`materials/assignment_brief.pdf` -- remains the actual graded deliverable
and lives at <https://github.com/raulrodriguezlr/B5-T1>; a full, unmodified
copy of its notebooks is kept here under
`notebooks/original_submission/` for reference and comparison.

This rewrite exists as a personal, standalone exercise in restructuring a
working data-science notebook project into a tested, typed, documented
Python package, and was not requested by nor is it intended to replace the
graded submission.

### Summary of changes

| File | Action | Reason |
|---|---|---|
| `src/utils/config.py` | Created | Centralised, typed configuration (was `src/miax_b5t1/config.py`, module-level only) |
| `src/utils/logging_config.py` | Created | `logging` replaces `print` throughout |
| `src/data/loader.py` | Created | Split price download/caching out of dataset construction |
| `src/data/dataset.py` | Created | English rewrite of `datos.py`: windows, labels, embargoed split, `TanhScaler` |
| `src/models/classifier.py` | Created | English rewrite of `modelo.py`: reference CNN, architecture search |
| `src/models/generators/base.py` | Created | New `BaseGenerator` interface; did not exist in the original |
| `src/models/generators/noise.py` | Created | Ported from `notebooks/03_generador_ruido.ipynb`; origin-distance bug fixed |
| `src/models/generators/gan.py` | Created | Ported from `notebooks/04_generador_cgan.ipynb` |
| `src/models/generators/vae.py` | Created | Ported from `notebooks/05_generador_cvae.ipynb` |
| `src/models/generators/diffusion.py` | Created | Ported from `notebooks/06_generador_diffusion.ipynb` |
| `src/evaluation/experiment.py` | Created | English rewrite of `experimento.py`; added `threshold_sensitivity` |
| `src/evaluation/statistics.py` | Created | Paired significance test and comparison table, split out of notebook 07 |
| `src/evaluation/mutual_information.py` | Created | Split out of notebook 07's mutual-information section |
| `src/visualization/plots.py` | Created | English rewrite of `graficos.py` |
| `main.py` | Created | Single CLI entry point; did not exist (the original ran through notebooks) |
| `tests/*.py` | Created | 40 unit tests; the original submission had none |
| `docs/architecture.md` | Created | Mermaid diagrams; did not exist |

### Key corrections

* **The noise generator's "distance to original" metric was measuring the
  wrong thing.** `notebooks/03_generador_ruido.ipynb`'s sigma-calibration
  table computed `X_s[:500] - X_real[:1]` -- the distance from 500 synthetic
  samples to a single, fixed real window, not to the window each sample was
  actually derived from. `src/models/generators/noise.py`'s
  `generate_with_provenance` returns the source index of every sample, so
  `calibrate_noise_intensity` measures the distance each synthetic sample
  actually travelled from its own origin
  (`tests/test_generators.py::test_noise_generator_distance_to_own_origin_is_small`
  guards against a regression).
* **The diffusion generator's textbook parameterisation does not converge
  on this data.** Predicting the added noise (the standard formulation)
  plateaued at a 0.80 training loss out of a 1.00 ceiling and produced
  saturated, unusable samples, because daily log returns have an
  autocorrelation near 0.019 -- close enough to white noise that the added
  Gaussian noise is not recoverable from the noisy observation. Predicting
  the clean window instead (`src/models/generators/diffusion.py`) reaches a
  0.34 training loss and samples cleanly. This is documented at length in
  the module's docstring, since it is the single most consequential design
  decision in the codebase.

### Improvements beyond the professor's solution

* **`threshold_sensitivity` (`src/evaluation/experiment.py`).** During the
  recorded correction lecture (`materials/class_transcription.txt`), the
  professor asked, live, whether the crisis-labelling threshold (fixed at
  the 10th percentile of drawdown) had been explored, noting that a less
  strict cut would produce more crisis windows to train on. This function
  answers that question directly: it recomputes the label and the resulting
  number of independent crisis episodes for a range of candidate positive
  rates, without touching the reference pipeline (`config.POSITIVE_RATE`
  stays the value every other result is reported against).
* **A finer synthetic-ratio grid.** The same lecture has the professor
  pointing out that several other teams found their best result between
  "no synthetics" and "as many synthetics as real data" -- a region the
  original `{0, 0.25, 0.5, 1, 2, 4}` grid samples coarsely.
  `config.SYNTHETIC_RATIOS` adds `0.10` and `0.75` without removing any of
  the original points, so previously computed results remain comparable at
  every ratio they already cover.
* **`BaseGenerator`.** Introducing a common `fit`/`generate`/`loss_history`
  interface (absent from the original notebooks, where each generator was
  free-standing code) is what lets `main.py`'s `train-generator` and
  `evaluate-generator` commands work identically for all four models,
  driven only by a `--generator <name>` argument.
* **A `Settings` dataclass alongside the module-level constants.** Frozen,
  so accidental cross-run mutation raises immediately instead of silently
  corrupting a later run; module-level constants are kept for direct,
  backward-compatible imports.

### Deviations from the professor's solution

None of the professor's own example notebooks
(`professor_notebooks/Taller_GANs.ipynb`,
`Taller_Gaussian_solution.ipynb`, `Taller_con_Datos_SP500_promedio.ipynb`)
targets this exact classification problem -- they are general-purpose
teaching examples (a GAN and a Gaussian generator applied to a *regression*
task on the same price universe, and a separate architecture comparison for
that regression task). Their generator architectures, training loops, and
evaluation protocol were used as a reference for good practice (batch
composition, discriminator freezing, and so on) rather than ported
directly, since the classification problem, its label, and its evaluation
protocol are specific to this assignment's brief and were designed for it
in the original submission.

## How to Run (reproduction checklist)

```bash
python -m venv .venv
.venv\Scripts\activate                  # or: source .venv/bin/activate
pip install -r requirements.txt
python main.py build-dataset
python main.py run-all --skip-diffusion # add the diffusion stage separately if time allows
pytest tests/ -v
```
