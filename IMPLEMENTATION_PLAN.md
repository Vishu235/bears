# BEARS — Implementation Plan & Demo Cheatsheet

> Run this file top-to-bottom. Every command is copy-pasteable as-is from the project root (`d:\PES\Semester 4\bears`).

---

## 0. One-Time Setup

### Activate the virtual environment
```powershell
# From project root
.\venv\Scripts\Activate.ps1
```

### Verify environment
```powershell
python --version          # should be 3.9.7
python -c "import torch; print(torch.__version__)"   # should be 1.13.0
python -c "import wandb; print(wandb.__version__)"   # should be 0.13.11
```

### All runs use this working directory
```powershell
cd "d:\PES\Semester 4\bears\XOR_MNIST"
```

---

## Phase 1 — XOR Toy Experiment

> **Purpose:** Prove the core claim on the simplest possible setting.  
> **What to watch:** Y-Acc stays 100% across ALL runs. C-Acc varies (low = RS present).  
> **Runtime:** ~30 seconds per run.

### 1.1 DPL (DeepProbLog)

```powershell
# Baseline — no mitigation (expect Y-Acc=100%, C-Acc low ~37%)
python -m example.xor_main --model dpl --seed 42

# + RECON (reconstruction decoder)
python -m example.xor_main --model dpl --rec --seed 42

# + Concept Supervision (4 labeled samples)
python -m example.xor_main --model dpl --csup 4 --seed 42

# + Entropy regularization
python -m example.xor_main --model dpl --entropy --seed 42

# + Disentangled encoder (one encoder per concept)
python -m example.xor_main --model dpl --disent --seed 42
```

### 1.2 SL (Semantic Loss)

```powershell
# Baseline
python -m example.xor_main --model sl --seed 42

# + RECON
python -m example.xor_main --model sl --rec --seed 42

# + Concept Supervision
python -m example.xor_main --model sl --csup 4 --seed 42

# + Entropy
python -m example.xor_main --model sl --entropy --seed 42
```

### 1.3 LTN (Logic Tensor Networks)

```powershell
# Baseline
python -m example.xor_main --model ltn --seed 42

# + RECON
python -m example.xor_main --model ltn --rec --seed 42

# + Concept Supervision
python -m example.xor_main --model ltn --csup 4 --seed 42

# + Entropy
python -m example.xor_main --model ltn --entropy --seed 42
```

### Phase 1 — Expected Output

| Model | Mitigation | Y-Acc | C-Acc | Interpretation |
|-------|-----------|-------|-------|----------------|
| DPL | None | 100% | ~37% | RS present — model found a shortcut |
| DPL | RECON | 100% | ~25–50% | Reconstruction alone may not fix RS |
| DPL | CSUP 4 | 100% | ~75–100% | Concept labels fix RS (costly) |
| DPL | Entropy | 100% | ~37–62% | Partial improvement |
| SL | None | 100% | ~37% | Same RS pattern |
| LTN | None | 100% | ~37% | Same RS pattern |

**Output files:** `XOR_MNIST/example/cfs/XOR-CF-<model><flags>.png` — confusion matrices

---

## Phase 2 — HalfMNIST (Main Benchmark)

> **Purpose:** Reproduce the primary table from the paper with real image data.  
> **What to watch:** BEARS should have highest H(C) and lowest ECE with same Y-Acc.  
> **Runtime:** ~30–60 min per model training run.

### 2.1 Train each model (Joint mode, best hyperparams)

```powershell
# DPL
python main.py --model mnistdpl --dataset halfmnist --load_best_args

# SL
python main.py --model mnistsl --dataset halfmnist --load_best_args

# LTN
python main.py --model mnistltn --dataset halfmnist --load_best_args
```

### 2.2 Post-hoc evaluation — all 4 uncertainty modes

Run each evaluation type after training (requires `--posthoc` flag and a saved checkpoint via `--checkin`):

```powershell
# --- DPL ---

# Normal (single model, no Bayesian)
python main.py --model mnistdpl --dataset halfmnist --posthoc --type normal \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args

# BEARS (ensemble + KL divergence — the proposed method)
python main.py --model mnistdpl --dataset halfmnist --posthoc --type bears \
  --n_ensembles 5 --lambda_h 10 \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args

# Deep Ensemble (ensemble without KL — ablation baseline)
python main.py --model mnistdpl --dataset halfmnist --posthoc --type ensemble \
  --n_ensembles 5 \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args

# MC Dropout (stochastic inference — Bayesian baseline)
python main.py --model mnistdpl --dataset halfmnist --posthoc --type mc_dropout \
  --n_ensembles 30 \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args

# --- Repeat above for mnistsl and mnistltn ---
```

> **Note:** Replace `<checkpoint_name>` with the actual `.pt` file saved by the training run.  
> Checkpoints are saved to `XOR_MNIST/data/ckpts/` by default.

### 2.3 Evaluate all modes in one shot

```powershell
# Runs Normal → BEARS → Ensemble → MC-Dropout sequentially
python main.py --model mnistdpl --dataset halfmnist --posthoc --evaluate_all \
  --n_ensembles 5 --lambda_h 10 \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args
```

### Phase 2 — Expected Output

| Model | Type | Y-Acc | C-Acc | H(C) | ECE |
|-------|------|-------|-------|------|-----|
| DPL | Normal | ~X% | ~X% | low | high |
| DPL | BEARS | ~X% | ~X% | **high** | **low** |
| DPL | Ensemble | ~X% | ~X% | medium | medium |
| DPL | MC-Drop | ~X% | ~X% | medium | medium |
| SL | Normal | ~X% | ~X% | low | high |
| SL | BEARS | ~X% | ~X% | **high** | **low** |
| LTN | Normal | ~X% | ~X% | low | high |
| LTN | BEARS | ~X% | ~X% | **high** | **low** |

**Output files:** `XOR_MNIST/dumps/<model_name>-seed_<N>-<type>-nens_<N>-ood_False-lambda_<N>.csv`

---

## Phase 3 — AddMNIST

> **Purpose:** Harder RS setting — 19 possible label values, more valid (d1, d2) pairs per label.  
> **Runtime:** ~60–90 min per model.

### 3.1 Train

```powershell
python main.py --model mnistdpl --dataset addmnist --load_best_args
python main.py --model mnistsl  --dataset addmnist --load_best_args
python main.py --model mnistltn --dataset addmnist --load_best_args
```

### 3.2 Evaluate with BEARS

```powershell
python main.py --model mnistdpl --dataset addmnist --posthoc --type bears \
  --n_ensembles 5 --lambda_h 10 \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args
```

---

## Phase 4 — Kandinsky Patterns

> **Purpose:** Visual reasoning with compositional concepts (shape + color).  
> **Data:** `XOR_MNIST/data/kand-3k.zip` — extract before running.  
> **Runtime:** ~2–3 hours.

### 4.1 Extract data

```powershell
cd "d:\PES\Semester 4\bears\XOR_MNIST\data"
Expand-Archive -Path kand-3k.zip -DestinationPath .
cd "d:\PES\Semester 4\bears\XOR_MNIST"
```

### 4.2 Train

```powershell
python main.py --model kanddpl --dataset kandinsky --load_best_args
```

### 4.3 Evaluate with BEARS

```powershell
python main.py --model kanddpl --dataset kandinsky --posthoc --type bears \
  --n_ensembles 5 --lambda_h 2 \
  --checkin data/ckpts/<checkpoint_name>.pt --load_best_args
```

---

## Quick Reference — All Flags

### Model choices (`--model`)
| Flag | Description |
|------|-------------|
| `mnistdpl` | DeepProbLog on MNIST |
| `mnistsl` | Semantic Loss on MNIST |
| `mnistltn` | Logic Tensor Networks on MNIST |
| `mnistdplrec` | DPL + reconstruction decoder |
| `mnistpcbmdpl` | DPL + Post-hoc Concept Bottleneck Model |
| `kanddpl` | DPL on Kandinsky |

### Dataset choices (`--dataset`)
| Flag | Description |
|------|-------------|
| `halfmnist` | Half-image MNIST |
| `addmnist` | Addition MNIST (d1 + d2 = Y) |
| `shortmnist` | Shortcut-injected MNIST |
| `restrictedmnist` | Multi-operation MNIST |
| `kandinsky` | Kandinsky visual patterns |

### Evaluation type (`--type`, used with `--posthoc`)
| Flag | Description |
|------|-------------|
| `normal` | Single model deterministic |
| `bears` | BEARS ensemble with KL divergence |
| `ensemble` | Plain deep ensemble (adversarial) |
| `mc_dropout` | MC Dropout (30 samples) |
| `laplace` | Laplace approximation (last layer) |

### Training mitigations (XOR toy only via `example/xor_main.py`)
| Flag | Description |
|------|-------------|
| `--rec` | Add reconstruction decoder (RECON) |
| `--csup N` | Concept supervision on N samples |
| `--entropy` | Entropy regularization |
| `--disent` | Disentangled per-concept encoders |

### Key hyperparameters
| Flag | Default | Description |
|------|---------|-------------|
| `--n_ensembles` | — | Number of ensemble members |
| `--lambda_h` | 10 | Weight of BEARS KL divergence term |
| `--seed` | 42 | Random seed |
| `--load_best_args` | — | Auto-loads tuned hyperparams from `exp_best_args.py` |
| `--checkin` | — | Path to saved checkpoint for posthoc evaluation |

---

## Output Locations

| Artifact | Location |
|----------|----------|
| Confusion matrix plots (XOR) | `XOR_MNIST/example/cfs/*.png` |
| Model checkpoints | `XOR_MNIST/data/ckpts/*.pt` |
| Evaluation CSVs | `XOR_MNIST/dumps/*.csv` |
| Kandinsky analysis | `XOR_MNIST/data/kand-analysis/` |

---

## Metrics to Report (Per Run)

| Metric | What it means | BEARS vs Normal |
|--------|---------------|-----------------|
| **Y-Acc** | % labels correct | Should be equal |
| **C-Acc** | % concepts correct (eval only, no training labels) | May decrease |
| **H(C)** | Mean entropy over concept predictions | BEARS > Normal |
| **ECE** | Expected Calibration Error (lower = better) | BEARS < Normal |
| **C-F1** | F1 over concept predictions | May change slightly |

---

## Summary Results Table (Fill In)

### XOR Toy

| Model | Mitigation | Y-Acc (%) | C-Acc (%) |
|-------|-----------|-----------|-----------|
| DPL | None | | |
| DPL | RECON | | |
| DPL | CSUP-4 | | |
| DPL | Entropy | | |
| SL | None | | |
| SL | RECON | | |
| SL | CSUP-4 | | |
| LTN | None | | |
| LTN | RECON | | |
| LTN | CSUP-4 | | |

### HalfMNIST

| Model | Type | Y-Acc | C-Acc | H(C) | ECE |
|-------|------|-------|-------|------|-----|
| DPL | Normal | | | | |
| DPL | BEARS | | | | |
| DPL | Ensemble | | | | |
| DPL | MC-Dropout | | | | |
| SL | Normal | | | | |
| SL | BEARS | | | | |
| LTN | Normal | | | | |
| LTN | BEARS | | | | |

### AddMNIST

| Model | Type | Y-Acc | C-Acc | H(C) | ECE |
|-------|------|-------|-------|------|-----|
| DPL | Normal | | | | |
| DPL | BEARS | | | | |
| SL | Normal | | | | |
| SL | BEARS | | | | |
| LTN | Normal | | | | |
| LTN | BEARS | | | | |

---

## Known Results (From Runs So Far)

| Model | Mitigation | Y-Acc | C-Acc | Notes |
|-------|-----------|-------|-------|-------|
| DPL | None | 100% | 37.5% | RS confirmed: C3 predicted as all-1s |
| DPL | RECON | 100% | 25.0% | RECON did not resolve RS on toy |
