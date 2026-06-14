# Internal Review 2 — Code Walkthrough & Execution Needs

> Slide-style notes explaining the repository code, how it maps to the experiments, and what is required to proceed with execution.

---

## Slide 1 — Title

**Internal Review 2: Code Walkthrough & Execution Plan**

---

## Slide 2 — Progress Recap

- Completed: literature survey, BEARS base paper analysis, repository exploration, environment scaffolding started.
- Goal for this review: explain the code mapping clearly and list concrete prerequisites to run experiments.

---

## Slide 3 — High-level Repo Map

- `XOR_MNIST/` — toy XOR experiments, MNIST variants, Kandinsky patterns (main reproduction area).
- `BDD_OIA/` — real-world driving benchmark and BDD-specific tooling.
- Top-level helpers: `IMPLEMENTATION_PLAN.md`, `PROJECT_OVERVIEW.md`, `requirements.txt`.

---

## Slide 4 — Key Entry Points (Code → Experiment)

- `XOR_MNIST/main.py` — train / post-hoc evaluation for MNIST, AddMNIST, Kandinsky experiments.
- `XOR_MNIST/example/xor_main.py` — XOR toy sanity check (fast, deterministic behaviour).
- `XOR_MNIST/exp_best_args.py` & `XOR_MNIST/experiments.py` — canonical hyperparameters and experiment templates.
- `BDD_OIA/main_bdd.py` — BDD-OIA pipeline (heavy-data real-world benchmark).

---

## Slide 5 — Important Code Areas

- Models: `XOR_MNIST/models/` (DPL, SL, LTN and variants).
- Data: `XOR_MNIST/datasets/`, `BDD_OIA/BDD/` (loaders and dataset configs).
- Backbones: `XOR_MNIST/backbones/` (encoders / decoders).
- Utilities: `XOR_MNIST/utils/` (training loop, losses, metrics, visualization).

---

## Slide 6 — How Code Implements BEARS (Conceptually)

- Neural encoder → concept probabilities P(C|X) (backbones + model heads).
- Symbolic constraint layer → model-specific logic (DPL, SL, LTN modules in `models/`).
- BEARS evaluation → ensemble post-hoc mode with KL separation (implemented in `utils/` and evaluation routines in `main.py`).
- Metrics & plots → entropy H(C), Y-Acc, C-Acc, ECE, confusion matrices (visualization utilities).

---

## Slide 7 — Short Walkthrough (Example: HalfMNIST)

- Where to look: `XOR_MNIST/datasets/` (loader), `XOR_MNIST/backbones/` (encoder), `XOR_MNIST/models/mnistdpl.py` (DPL model variant), `XOR_MNIST/utils/train.py` and `XOR_MNIST/utils/test.py` (training/eval loop), `XOR_MNIST/exp_best_args.py` (recommended hyperparams).
- Expected artifacts: model checkpoint files, per-run CSV summaries in `XOR_MNIST/dumps/`, and plots in `XOR_MNIST/dumps/plots/`.

---

## Slide 8 — Execution Plan (What I Will Run Next)

- Step 1: XOR toy sanity checks and output inspection (fast).
- Step 2: HalfMNIST — reproduce main tables and BEARS vs baselines (moderate time).
- Step 3: AddMNIST & Kandinsky if results stable (longer runs).
- Optional: BDD-OIA after data and pretrained detectors are available.

---

## Slide 9 — XOR Experiment: What Was Run

Three configurations were executed on the XOR toy benchmark using the DPL model:

| Run | Command flag | What it tests |
|---|---|---|
| Baseline | (none) | Model with no mitigation — pure RS exposure |
| RECON | `--rec` | Reconstruction decoder as unsupervised mitigation |
| CSUP | `--csup 4` | 4 ground-truth concept labels as supervised mitigation |

Confusion matrix outputs saved to `XOR_MNIST/example/cfs/`.

---

## Slide 10 — XOR Results: Baseline DPL

**File:** `example/cfs/XOR-CF-dpl.png`

- The 8×8 concept confusion matrix (left panel) is **scattered and off-diagonal** — the model has learned a Reasoning Shortcut. It consistently predicts a particular concept assignment that satisfies the XOR rule, but it is not the ground-truth assignment.
- The three individual concept matrices (C1, C2, C3) show **mixed predictions** — the model cannot reliably distinguish 0 from 1 for each concept.
- Task accuracy (Y-Acc) is high — the symbolic rule is satisfied — but concept accuracy (C-Acc) is poor.
- This directly demonstrates the core problem: high task accuracy masks fundamentally wrong concept learning.

---

## Slide 11 — XOR Results: DPL + RECON (Unsupervised Mitigation)

**File:** `example/cfs/XOR-CF-dpl-rec.png`

- The 8×8 matrix is more concentrated compared to baseline — fewer off-diagonal predictions.
- However the individual concept matrices still show asymmetric behavior: C1 and C2 show some structure, but C3 is predicted as 0 in almost all cases (the matrix is nearly all blue except a single yellow block).
- RECON adds a reconstruction decoder that forces the encoder to preserve input information. This partially reduces RSs without requiring any concept labels.
- **Takeaway:** RECON helps but does not fully break the shortcut. Concept predictions are less wrong but still biased.

---

## Slide 12 — XOR Results: DPL + CSUP (4 Concept Labels)

**File:** `example/cfs/XOR-CF-dpl-csup_4.png`

- The 8×8 concept confusion matrix is **near-diagonal** — most inputs are mapped to their correct concept triplet.
- All three individual concept matrices (C1, C2, C3) show a clear diagonal structure: the model correctly predicts 0 or 1 for each concept.
- Just **4 labeled concept examples** is enough to break the RS entirely and force the model onto the correct semantic interpretation.
- **Takeaway:** Supervised concept labels are highly effective — but expensive to collect in real settings. This motivates the need for BEARS, which aims to achieve RS-awareness without any concept labels.

---

## Slide 13 — XOR Summary: What the Three Runs Show Together

| Configuration | Concept CF | C-Acc | Y-Acc | Concept labels needed |
|---|---|---|---|---|
| Baseline DPL | Scattered / off-diagonal | Low | High | None |
| DPL + RECON | Partially concentrated | Medium | High | None |
| DPL + CSUP(4) | Near-diagonal | High | High | 4 samples |

- The XOR experiment confirms the paper’s core claim: NeSy models learn RSs by default.
- Unsupervised mitigation (RECON) partially helps.
- A small amount of supervision fully solves it — but acquiring labels at scale is the real-world bottleneck.
- BEARS (to be run in Phase 2) targets this gap: RS-awareness without concept labels, via ensemble calibration.

---

## Slide 14 — Plan for the Next 3 Weeks

**Week 1 (this week → next review checkpoint):**
- Run HalfMNIST experiments: baseline DPL (Normal mode) + BEARS post-hoc evaluation.
- Collect Y-Acc, C-Acc, H(C) for at least the DPL model on HalfMNIST.
- Deliverable: a small result table with Normal vs BEARS comparison for DPL on HalfMNIST.

**Week 2:**
- Extend HalfMNIST runs to SL and LTN models.
- Run Deep Ensemble and MC-Dropout evaluation modes alongside BEARS for comparison.
- Begin AddMNIST experiments (same structure, higher RS ambiguity).

**Week 3:**
- Run Kandinsky Patterns experiments (visual reasoning, compositional concepts).
- Consolidate all results into a summary table (dataset × model × evaluation type).
- Identify any deviations from the paper’s reported numbers and document them.

---

## Slide 15 — What Will Be Ready by Next Week

By the next checkpoint, the following will be complete:

1. **HalfMNIST DPL baseline run** — Y-Acc, C-Acc, H(C), ECE numbers for Normal mode.
2. **HalfMNIST DPL BEARS run** — same metrics under BEARS post-hoc evaluation (5 ensemble members).
3. **A partial reproduction table** comparing Normal vs BEARS for DPL on HalfMNIST — the first concrete check against the paper’s Table 1.
4. **Environment confirmed stable** for the main MNIST pipeline (dependencies resolved, data auto-downloaded, dumps/ populated).

This gives a clear answer to the key question: does the reproduced BEARS show higher H(C) than the Normal baseline while preserving Y-Acc?

---

## Slide 16 — Appendix: Quick File Reference

- `XOR_MNIST/main.py`, `XOR_MNIST/example/xor_main.py`, `XOR_MNIST/models/`, `XOR_MNIST/datasets/`, `XOR_MNIST/utils/`.
- `BDD_OIA/main_bdd.py`, `BDD_OIA/BDD/`, and BDD preprocessing scripts.
- XOR confusion matrix outputs: `XOR_MNIST/example/cfs/XOR-CF-dpl.png`, `XOR-CF-dpl-rec.png`, `XOR-CF-dpl-csup_4.png`.
