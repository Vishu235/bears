# BEARS Project Presentation Plan

> A slide-ready markdown outline for presenting how I will spend the next couple of weeks reproducing the BEARS paper and running the codebase.

---

## Slide 1 — Title

**Project:** Reproducing BEARS: BE Aware of Reasoning Shortcuts  
**Goal:** Understand the codebase, reproduce the base paper results, and prepare a clear summary of the experiments and outcomes.

---

## Slide 2 — What This Project Is About

- BEARS studies reasoning shortcuts in neuro-symbolic models.
- The main idea is to keep task accuracy stable while making concept predictions more uncertainty-aware.
- The codebase contains multiple experiment families, from toy XOR examples to real-world BDD-OIA experiments.
- My focus for the next couple of weeks is to reproduce the base paper on the most informative benchmarks first.

---

## Slide 3 — Codebase Map

### Main folders

- `XOR_MNIST/` — the primary reproduction area for toy and MNIST-based experiments.
- `BDD_OIA/` — the larger autonomous-driving benchmark and related model code.

### Key support files

- `README.md` — high-level project summary, datasets, and code structure.
- `PROJECT_OVERVIEW.md` — paper context, dataset descriptions, and reproduction strategy.
- `IMPLEMENTATION_PLAN.md` — detailed run-oriented implementation reference.
- `requirements.txt` and `requirements.dev.txt` — environment dependencies.
- `Makefile` — project helper commands and workflow shortcuts.

---

## Slide 4 — Main Entry Points

### XOR / MNIST / Kandinsky pipeline

- `XOR_MNIST/main.py` — main training and evaluation entry point.
- `XOR_MNIST/example/xor_main.py` — small XOR demonstration used as a sanity check.
- `XOR_MNIST/experiments.py` — experiment definitions and run configurations.
- `XOR_MNIST/exp_best_args.py` — best hyperparameter settings per dataset and model.

### BDD-OIA pipeline

- `BDD_OIA/main_bdd.py` — main script for the real-world driving benchmark.
- `BDD_OIA/server.py` — related serving / orchestration support.

---

## Slide 5 — Important File Groups

### Models

- `XOR_MNIST/models/` — DPL, SL, LTN, and related MNIST/Kandinsky model variants.
- `BDD_OIA/models.py`, `BDD_OIA/DPL/`, `BDD_OIA/SENN/` — model implementations for the BDD setting.

### Datasets and preprocessing

- `XOR_MNIST/datasets/` — HalfMNIST, AddMNIST, ShortcutMNIST, RestrictedMNIST, Kandinsky, and utilities.
- `XOR_MNIST/backbones/` — encoders and decoders used by the image experiments.
- `BDD_OIA/BDD/` — dataset configuration and data handling for the driving benchmark.

### Training, metrics, and visualization

- `XOR_MNIST/utils/` — training loop, losses, metrics, checkpointing, and visualization helpers.
- `BDD_OIA/trainers_BDD.py`, `BDD_OIA/testers_BDD.py`, `BDD_OIA/visualization.py` — BDD training, evaluation, and plotting.

---

## Slide 6 — Datasets I Will Use

### 1. XOR-MNIST toy setting

- Purpose: validate the core reasoning-shortcut behavior in the smallest possible setting.
- Value: fastest sanity check for the model behavior and concept uncertainty.

### 2. HalfMNIST

- Purpose: main benchmark for reproducing the paper’s core claim on image data.
- Value: good balance between simplicity and realism.

### 3. AddMNIST

- Purpose: harder benchmark with more valid concept combinations for the same label.
- Value: useful for checking whether the method still works when shortcut ambiguity increases.

### 4. Kandinsky patterns

- Purpose: compositional visual reasoning benchmark.
- Value: a stronger test of concept-based reasoning beyond digits.

### 5. BDD-OIA

- Purpose: real-world driving benchmark.
- Value: important for completeness, but likely later in the timeline because it is heavier to set up and interpret.

---

## Slide 7 — What I Will Reproduce

- Baseline behavior on the XOR toy problem.
- Main comparison on HalfMNIST across the supported model families.
- BEARS versus other uncertainty modes such as normal inference, plain ensemble, and MC dropout.
- Key ablations and mitigation strategies such as reconstruction, entropy regularization, concept supervision, and disentangled encoders.
- If time allows, the stronger benchmark cases: AddMNIST, Kandinsky, and then BDD-OIA.

---

## Slide 8 — Deliverables

### Technical deliverables

- A working reproduction of the core BEARS experiments.
- Saved checkpoints and result summaries for the selected runs.
- Comparison tables for task accuracy, concept accuracy, entropy, and calibration metrics.
- Plots and confusion matrices that visually show the reasoning-shortcut effect.

### Presentation deliverables

- A short slide deck explaining the project structure and experiment plan.
- A concise summary of what was reproduced, what matched the paper, and what still needs work.
- A clear explanation of which datasets and model families were prioritized and why.

---

## Slide 9 — Two-Week Work Plan

### Week 1: Understand and verify

- Read the project structure and identify the important experiment files.
- Confirm the environment and dependencies.
- Run the XOR toy benchmark first to validate the reasoning-shortcut behavior.
- Inspect the outputs and confirm that the metrics and plots match the expected pattern.
- Move to HalfMNIST and verify the main model pipeline.

### Week 2: Reproduce and compare

- Run the main model comparisons on HalfMNIST.
- Collect the results for BEARS and the baseline uncertainty methods.
- Test the main mitigation strategies to see how they affect concept uncertainty.
- Extend to AddMNIST or Kandinsky if the core benchmarks are stable.
- Prepare the final summary of findings, limitations, and next steps.

---

## Slide 10 — Success Criteria

- The code runs cleanly for the key benchmarks.
- The toy XOR example shows the expected reasoning-shortcut pattern.
- HalfMNIST reproduces the main comparison between BEARS and baseline uncertainty methods.
- The outputs clearly show where BEARS increases concept uncertainty without harming task accuracy.
- The final summary is understandable as a presentation, not just as a run log.

---

## Slide 11 — Final Message

This work will turn the repository into a reproducible story:

1. Start with the code structure.
2. Verify the toy benchmark.
3. Reproduce the main paper benchmark.
4. Collect the plots and tables.
5. Present the results as a clear two-week research plan.

---

## Slide 12 — Appendix: File Summary

- `XOR_MNIST/main.py` — main training and evaluation hub.
- `XOR_MNIST/example/` — toy XOR reproduction and sanity checks.
- `XOR_MNIST/models/` — model definitions for DPL, SL, LTN, and related variants.
- `XOR_MNIST/datasets/` — dataset loaders for the MNIST-based benchmarks.
- `XOR_MNIST/utils/` — training, metrics, checkpoints, and visualization.
- `BDD_OIA/main_bdd.py` — real-world benchmark entry point.
- `BDD_OIA/DPL/`, `BDD_OIA/SENN/`, `BDD_OIA/models.py` — BDD-specific model implementations.
- `BDD_OIA/trainers_BDD.py`, `BDD_OIA/testers_BDD.py`, `BDD_OIA/visualization.py` — BDD training and reporting helpers.
