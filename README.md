# BEARS — BE Aware of Reasoning Shortcuts

Reproduction of the BEARS method for detecting and mitigating Reasoning Shortcuts in Neuro-Symbolic models.

## Abstract

Neuro-Symbolic (NeSy) predictors that conform to symbolic knowledge can be affected by Reasoning Shortcuts (RSs): they learn concepts consistent with the symbolic knowledge by exploiting unintended semantics. RSs compromise reliability and generalization and are linked to NeSy models being overconfident about predicted concepts.

BEARS (BE Aware of Reasoning Shortcuts) is an ensembling technique that calibrates the model's concept-level confidence without compromising prediction accuracy, encouraging NeSy architectures to be uncertain about concepts affected by RSs.

## Installation

Requires **Python 3.9**.

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# or: source venv/bin/activate  # Linux/Mac

pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

### BDD-OIA

A dashcam dataset for autonomous driving predictions, annotated with concept-level entities (e.g. "road is clear"). Preprocessed with a pretrained Faster-RCNN on BDD-100k, producing 2048-dim embeddings (`bdd_2048.zip`).

Original dataset: https://twizwei.github.io/bddoia_project/

### MNIST-Even-Odd

A variant of MNIST-Addition using only even/odd digit combinations (e.g. 0+6=6, 2+8=10). Training: 6720 samples, validation: 1920, test: 960, OOD test: 5040 samples.

### MNIST-Half

A biased MNIST-Addition variant using digits 0–4 only. Training: 2940 samples, validation: 840, test: 420, OOD test: 1080 samples.

### Kandinsky Patterns

Visual reasoning dataset with geometric figures. Each figure has one of three colors (red, blue, yellow) and one of three shapes (square, circle, triangle). The task is to predict the pattern of a third image given two images sharing a common pattern.

## Code Structure

```
bears/
├── XOR_MNIST/           # XOR, MNIST-Addition, and Kandinsky experiments
│   ├── backbones/       # Neural network architectures
│   ├── datasets/        # Dataset loaders (MNIST variants, Kandinsky)
│   ├── example/         # Standalone XOR logic demo
│   ├── models/          # DPL, SL, LTN model definitions
│   ├── utils/           # Training loop, losses, metrics, BEARS ensembling
│   ├── main.py          # Entry point
│   ├── exp_best_args.py # Best hyperparameters per task
│   └── experiments.py   # Experiment configuration templates
│
└── BDD_OIA/             # Autonomous driving experiments
    ├── BDD/             # Dataset config and loader
    ├── DPL/             # DeepProbLog models for BDD
    ├── SENN/            # Self-Explaining Neural Network
    ├── main_bdd.py      # Entry point
    └── run_bdd.sh       # Launch script
```

## Running Experiments

### Quick smoke test (no data needed)
```bash
cd XOR_MNIST/example
python xor_main.py
```

### MNIST-Addition with DPL
```bash
cd XOR_MNIST
python main.py --model dpl --dataset addmnist --task addition --c_sup 0.1
```

### Post-hoc evaluation with BEARS
```bash
python main.py --model dpl --dataset addmnist --posthoc --type bears --n-models 5
```

### All evaluation strategies at once
```bash
python main.py --model dpl --dataset addmnist --posthoc --evaluate-all
```

### BDD-OIA
```bash
cd BDD_OIA
bash run_bdd.sh
```

## Key Arguments

| Argument | Description |
|---|---|
| `--dataset` | `addmnist`, `halfmnist`, `restrictedmnist`, `kandinsky`, `bddoia` |
| `--task` | `addition`, `product`, `multiop` |
| `--model` | `dpl`, `sl`, `ltn`, `kanddpl` |
| `--c_sup` | Concept supervision fraction (0.0 = none, 1.0 = full) |
| `--which_c` | List of concept indices to supervise, e.g. `[1,2]` |
| `--posthoc` | Enable post-hoc evaluation |
| `--type` | Evaluation method: `bears`, `ensemble`, `mcdropout`, `laplace`, `frequentist` |
| `--n-models` | Number of ensemble members (default: 5) |
| `--entropy` | Add entropy regularization penalty |
| `--wandb` | W&B project name for experiment tracking |
| `--checkin` | Path to load checkpoint from |
| `--checkout` | Path to save checkpoint to |
| `--validate` | Use validation set instead of test set |
