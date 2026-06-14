# BEARS — Project Overview & Reproduction Plan

> **Base Paper:** BEARS Make Neuro-Symbolic Models Aware Of Their Reasoning Shortcuts  
> **ArXiv:** https://arxiv.org/abs/2402.12240  
> **Authors:** Emanuele Marconato, Samuele Bortolotti, Emile van Krieken, Antonio Vergari, Andrea Passerini, Stefano Teso  
> **Affiliations:** University of Trento & University of Edinburgh  
> **Released:** February 2024

---

## 1. What This Paper Is About

### 1.1 The Core Problem: Reasoning Shortcuts

Neuro-Symbolic (NeSy) AI systems combine neural networks (for perception) with symbolic reasoning (for logic/constraints). A typical setup looks like this:

```
Image/Input → [Neural Encoder] → Concepts (C) → [Symbolic Reasoner] → Label (Y)
```

The symbolic reasoner enforces a known logical rule. For example:

- "Label Y = 1 if and only if concept C1 XOR C2 XOR C3 = 1"
- "Car should brake if pedestrian is in road AND traffic light is red"

The neural encoder is trained to predict concepts C from raw input. The problem is that the symbolic rule does not uniquely determine what C should be. Multiple assignments of C can satisfy the rule and still predict Y correctly. This is a **Reasoning Shortcut (RS)**: the model finds semantically incorrect but logically valid concepts.

**Example:** In the XOR task with 3 bits, if the true answer is Y=1, there are 4 valid concept assignments: (0,0,1), (0,1,0), (1,0,0), (1,1,1). The model may learn any one of them consistently — even if it does not match the true underlying concept.

### 1.2 Why RSs Are Harmful

- The model generalizes incorrectly to new distributions
- Concepts lose interpretability (a concept node labeled "pedestrian" may really encode something else)
- The only known trustworthy fix — collecting dense concept-level supervision — is expensive
- RSs are hard to detect because task accuracy (Y-Acc) stays high

### 1.3 The BEARS Proposal

Rather than eliminating RSs (which requires costly labels), BEARS makes the model **aware** of the ambiguity it has learned. The idea: a model that learned an RS should express **uncertainty** about its concept predictions, because multiple concept assignments are consistent with the symbolic rule.

**BEARS = BE Aware of Reasoning Shortcuts**

BEARS is an **ensemble calibration technique** derived from three desiderata:

1. **RS-awareness:** Concept predictions should be uncertain when the input is consistent with multiple RSs
2. **Accuracy preservation:** Task prediction accuracy must not be compromised
3. **Label efficiency:** No dense concept-level supervision should be required

The mechanism: train an ensemble of NeSy models with different random seeds. The ensemble is trained with a KL-divergence term that encourages each member to disagree with the others at the concept level (while still agreeing on the label). This spreads out concept probability mass across the ambiguous RSs, making the model genuinely uncertain about which concept assignment is correct.

---

## 2. The Datasets

### 2.1 XOR-MNIST (Toy Benchmark)

**Location:** `XOR_MNIST/example/`  
**Entry point:** `python -m example.xor_main --model dpl`

**What it is:**  
The simplest possible NeSy benchmark. There are 3 binary concepts C1, C2, C3. The symbolic rule is:

```
Y = C1 XOR C2 XOR C3
```

There are 8 possible input combinations (all binary triplets). This is a fully controlled setting — no images, just binary vectors as input.

**Why it is useful:**  
This dataset is designed so that RSs are analytically predictable. Because there are 4 valid concept assignments for each label (e.g., Y=0 can come from (0,0,0), (0,1,1), (1,0,1), (1,1,0)), a model without any concept supervision will learn one arbitrarily. The confusion matrices and accuracy numbers directly show whether a model has learned a RS. Since this dataset is tiny and requires no file I/O, it is the fastest way to verify that the code and concepts are working.

**What to show:**  
A model trained with no concept supervision (baseline) will show ~100% Y-Acc but low or inconsistent C-Acc. BEARS should increase concept-level uncertainty (H(C)) without dropping Y-Acc.

---

### 2.2 HalfMNIST

**Location:** `XOR_MNIST/` (main pipeline)  
**Entry point:** `python main.py --model mnistdpl --dataset halfmnist`

**What it is:**  
MNIST digits are split in half. The model sees only the top half of a digit image and must predict the digit. Concepts are individual digit identities. The task combines two digits' predictions under a symbolic rule.

**Why it is useful:**  
This is the primary benchmark in the paper because it introduces visual perception (real MNIST images) while keeping the concept space well-defined (10 digit classes per image slot). The deliberate use of half-images means the visual signal is ambiguous — multiple digit identities can be plausible from the top half alone — which naturally induces RSs. It is the most important table to reproduce.

**What to show:**  
Compare four evaluation modes (Normal, BEARS, Deep Ensemble, MC Dropout) on three models (DPL, SL, LTN). Key metrics: Y-Acc (should stay ~equal), H(C) (BEARS should be highest), ECE (BEARS should be lowest).

---

### 2.3 AddMNIST

**Location:** `XOR_MNIST/`  
**Entry point:** `python main.py --model mnistdpl --dataset addmnist`

**What it is:**  
Two MNIST digit images are shown side by side. The label Y is the sum of the two digits (Y = d1 + d2, range 0–18). Concepts are the individual digit identities (10 classes each).

**Why it is useful:**  
The addition rule is deterministic but many (d1, d2) pairs can sum to the same Y. For example, Y=5 can come from (0,5), (1,4), (2,3), (3,2), (4,1), (5,0). This creates natural RSs — any of these digit assignments is consistent with Y=5. The model can latch onto any shortcut that satisfies the sum constraint. This dataset is harder than HalfMNIST and has more RSs, so BEARS' RS-awareness benefit should be more pronounced.

**Symbolic rule:** `Y = C1 + C2` where C1, C2 ∈ {0, ..., 9}

---

### 2.4 ShortcutMNIST

**Location:** `XOR_MNIST/`  
**Entry point:** `python main.py --model mnistdpl --dataset shortmnist`

**What it is:**  
A modified version of MNIST where a deliberate visual shortcut is injected. Specific digit classes (e.g., 4 and 9) have a spurious visual feature added that strongly correlates with the label in the training set but not at test time.

**Why it is useful:**  
This dataset tests whether BEARS can detect _deliberately planted_ RSs, not just naturally occurring ones. If the model learns the shortcut feature instead of the true digit identity, BEARS should express higher uncertainty on the concept corresponding to the shortcut-affected class. This is a controlled ablation that validates the RS-awareness claim under a known ground truth shortcut.

---

### 2.5 RestrictedMNIST (Multi-Operation)

**Location:** `XOR_MNIST/`  
**Entry point:** `python main.py --model mnistltn --dataset restrictedmnist --task multiop`

**What it is:**  
A more complex MNIST benchmark where multiple arithmetic operations are combined (not just addition). The symbolic rule involves multiple concepts and multiple operations, increasing the number of valid RS assignments.

**Why it is useful:**  
Tests BEARS at larger concept spaces and more complex symbolic rules. More RSs means more ambiguity, which should amplify the H(C) improvement BEARS achieves.

---

### 2.6 Kandinsky Patterns

**Location:** `XOR_MNIST/`  
**Entry point:** `python main.py --model kanddpl --dataset kandinsky`  
**Data:** `XOR_MNIST/data/kand-3k.zip` (already present)

**What it is:**  
Images contain geometric shapes with specific properties (color and shape type). The label depends on whether the shapes follow a particular visual pattern (e.g., "all three objects have the same shape"). Concepts are shape properties: {shape type (3 classes), color (3 classes)} per object.

**Why it is useful:**  
This is a visual reasoning benchmark where concepts are genuinely compositional. The Kandinsky dataset is specifically designed so that multiple valid concept assignments exist for borderline images, making it a rich RS testbed. The dataset comes with 3000 images (`kand-3k.zip`) and is self-contained.

**Symbolic rule:** `Y = 1` if the Kandinsky pattern is satisfied (defined by shape/color conditions on multiple objects).

---

### 2.7 BDD-OIA (Real-World Benchmark)

**Location:** `BDD_OIA/`  
**Entry point:** `python BDD_OIA/main_bdd.py`  
**Data:** External download required (~10GB)

**What it is:**  
A real-world autonomous driving dataset. Given a dashcam frame, the model must predict the correct driving action (go forward, turn left, turn right, stop) based on 21 visual concepts (traffic light state, pedestrian presence, vehicle proximity, etc.). This is based on the Berkeley DeepDrive OIA dataset.

**Why it is useful:**  
This is the paper's real-world validation. Autonomous driving is a safety-critical domain where RSs are genuinely dangerous — a model that learns a shortcut (e.g., associating "clear road" with "turn right" spuriously) could cause accidents. BEARS' RS-awareness in this setting is the strongest motivation for the method.

**Why we skip it for now:**  
The dataset requires registering and downloading ~10GB of annotated video frames, plus a pretrained Faster-RCNN backbone for object detection. This is a significant infrastructure dependency that we defer until after the MNIST benchmarks are confirmed working.

---

## 3. The Models

Three NeSy architectures are evaluated, each representing a different paradigm of integrating neural and symbolic computation:

### 3.1 DPL — DeepProbLog

**Code:** `XOR_MNIST/models/mnistdpl.py`, `XOR_MNIST/example/dpl_models.py`

DPL uses probabilistic logic programming. The neural encoder produces concept probabilities P(C|X). These are fed into a probabilistic logic program that computes P(Y|C) using exact probabilistic inference over all possible concept assignments (sum over all worlds). The loss is negative log-likelihood of Y.

**Key property:** Concepts are treated as discrete probabilistic events. The symbolic rule is encoded as a ProbLog program.

### 3.2 SL — Semantic Loss

**Code:** `XOR_MNIST/models/mnistsl.py`

SL adds a regularization term to the standard cross-entropy loss. This term penalizes concept predictions that are inconsistent with the symbolic rule. Rather than using probabilistic inference, it uses a weighted satisfiability score.

**Key property:** The symbolic rule is compiled into a weighted formula. No explicit world enumeration is needed, making it faster but less principled than DPL.

### 3.3 LTN — Logic Tensor Networks

**Code:** `XOR_MNIST/models/mnistltn.py`

LTN encodes symbolic rules as differentiable fuzzy logic expressions. The neural encoder produces continuous concept values (in [0,1]). The symbolic rule is written as a fuzzy logical formula and its truth value is maximized during training via backpropagation.

**Key property:** Concepts are treated as continuous, not discrete. Fuzzy logic allows differentiable enforcement of constraints without explicit world enumeration.

---

## 4. Evaluation Modes

For each model and dataset, there are four ways to produce concept predictions (controlled via `--type`):

| Mode          | Code flag           | What it does                                                                  |
| ------------- | ------------------- | ----------------------------------------------------------------------------- |
| Normal        | `--type normal`     | Single model, deterministic inference                                         |
| BEARS         | `--type bears`      | Ensemble with KL divergence pushing members apart at concept level            |
| Deep Ensemble | `--type ensemble`   | Ensemble without KL divergence (standard deep ensemble, adversarial training) |
| MC Dropout    | `--type mc_dropout` | Single model, dropout active at test time for stochastic samples              |
| Laplace       | `--type laplace`    | Laplace approximation on the last layer                                       |

BEARS = Deep Ensemble + the KL divergence separation term (controlled by `--lambda_h`).

---

## 5. Mitigation Strategies Compared

Beyond the evaluation mode, the paper also tests various _training-time_ mitigations that can be combined:

| Flag        | Name                | What it does                                                                    | Cost                       |
| ----------- | ------------------- | ------------------------------------------------------------------------------- | -------------------------- |
| (none)      | Baseline            | Standard NeSy training                                                          | Free                       |
| `--rec`     | RECON               | Adds a reconstruction decoder; forces disentangled, locally consistent concepts | No concept labels          |
| `--entropy` | Entropy Reg         | Adds a Shannon entropy term to encourage concept uncertainty during training    | No concept labels          |
| `--csup N`  | Concept Supervision | Provides ground-truth concept labels for N training samples                     | Requires N labeled samples |
| `--disent`  | Disentanglement     | Uses a separate encoder per concept (no shared weights)                         | No concept labels          |

The key comparison BEARS makes: it achieves RS-awareness comparable to CSUP _without any concept labels_.

---

## 6. Key Metrics

| Metric                     | Symbol | What it measures                                                                 | BEARS effect                           |
| -------------------------- | ------ | -------------------------------------------------------------------------------- | -------------------------------------- |
| Task Accuracy              | Y-Acc  | % of labels predicted correctly                                                  | Must stay the same                     |
| Concept Accuracy           | C-Acc  | % of concepts predicted correctly (requires labels for evaluation, not training) | May decrease (model becomes uncertain) |
| Mean Concept Entropy       | H(C)   | Average entropy of concept probability distributions                             | Should increase (more RS-aware)        |
| Expected Calibration Error | ECE    | How well confidence scores match empirical accuracy                              | Should decrease                        |
| Concept F1                 | C-F1   | F1 score over concept predictions                                                | May decrease slightly                  |

---

## 7. Reproduction Plan

### Phase 1: Environment & Sanity Check (XOR Toy)

**Goal:** Confirm the code runs end-to-end and matches the paper's qualitative claim on the toy problem.

**Steps:**

1. Verify Python environment: Python 3.9+, PyTorch, wandb, warmup-scheduler, sklearn
2. Run baseline DPL on XOR (no mitigation):
   ```bash
   cd XOR_MNIST
   python -m example.xor_main --model dpl --seed 42
   ```
3. Run with RECON:
   ```bash
   python -m example.xor_main --model dpl --rec --seed 42
   ```
4. Run with concept supervision:
   ```bash
   python -m example.xor_main --model dpl --csup 4 --seed 42
   ```
5. Repeat for `--model sl` and `--model ltn`
6. Check: `example/cfs/` should contain confusion matrix images

**Expected output:**

- Baseline: Y-Acc ~100%, C-Acc varies (RS present)
- CSUP: Y-Acc ~100%, C-Acc higher (RS mitigated, but needs labels)
- Entropy/RECON: Y-Acc ~100%, C-Acc moderate (partial mitigation)

---

### Phase 2: HalfMNIST — Main Benchmark

**Goal:** Reproduce Table 1 (or equivalent) from the paper comparing Normal, BEARS, Deep Ensemble, and MC Dropout.

**Steps:**

1. Train and evaluate baseline DPL on HalfMNIST:
   ```bash
   cd XOR_MNIST
   python main.py --model mnistdpl --dataset halfmnist --load_best_args
   ```
2. Evaluate with BEARS:
   ```bash
   python main.py --model mnistdpl --dataset halfmnist --posthoc --type bears --n_ensembles 5 --load_best_args
   ```
3. Evaluate with plain Deep Ensemble:
   ```bash
   python main.py --model mnistdpl --dataset halfmnist --posthoc --type ensemble --n_ensembles 5 --load_best_args
   ```
4. Evaluate with MC Dropout:
   ```bash
   python main.py --model mnistdpl --dataset halfmnist --posthoc --type mc_dropout --load_best_args
   ```
5. Repeat for `mnistsl` and `mnistltn`

**Expected output:**

- BEARS > Deep Ensemble > Normal in H(C)
- Y-Acc stays flat across all methods
- BEARS has lower ECE than Normal

---

### Phase 3: AddMNIST

**Goal:** Confirm BEARS generalizes to a harder RS setting (more valid assignments per label).

**Steps:**

Same structure as Phase 2 but with `--dataset addmnist`. The higher RS ambiguity (19 possible label values, many (d1,d2) pairs per label) should make the H(C) gap larger.

---

### Phase 4: Kandinsky Patterns

**Goal:** Validate on a visual reasoning benchmark with compositional concepts (shape + color).

**Steps:**

1. Extract `XOR_MNIST/data/kand-3k.zip` to `XOR_MNIST/data/`
2. Run:
   ```bash
   python main.py --model kanddpl --dataset kandinsky --load_best_args
   ```
3. Evaluate with BEARS posthoc.

**Note:** The Kandinsky evaluation also includes an active learning phase (`--active_learning`) where BEARS uncertainty is used to select which samples to annotate. This is a secondary contribution of the paper.

---

### Phase 5: Results Consolidation

**Goal:** Collect numbers into a summary table for the project report.

For each configuration (dataset × model × evaluation type), record:

| Dataset   | Model | Type     | Y-Acc | C-Acc | H(C) | ECE |
| --------- | ----- | -------- | ----- | ----- | ---- | --- |
| HalfMNIST | DPL   | Normal   |       |       |      |     |
| HalfMNIST | DPL   | BEARS    |       |       |      |     |
| HalfMNIST | DPL   | Ensemble |       |       |      |     |
| HalfMNIST | DPL   | MC-Drop  |       |       |      |     |
| HalfMNIST | SL    | Normal   |       |       |      |     |
| ...       | ...   | ...      |       |       |      |     |

---

## 8. Novelty Direction (After Reproduction)

Once base results are confirmed, the following directions are available to extend the work:

1. **BEARS + Active Learning Integration:** BEARS already facilitates annotation (Phase 4 mentions this). Formalizing the active learning loop (select high-uncertainty samples → annotate → retrain → re-evaluate) and measuring annotation efficiency is a concrete extension.

2. **Hierarchical or Continuous Concepts:** BEARS currently operates on discrete, flat concept spaces. Extending it to hierarchical concepts (e.g., "animal → dog → golden retriever") or continuous concept representations is a natural generalization.

3. **BDD-OIA Evaluation:** Running BEARS on the real-world autonomous driving dataset with a proper evaluation of safety-critical RS cases strengthens the real-world applicability claim.

4. **Alternative Ensemble Strategies:** The BEARS KL term encourages diversity in concept space. Alternatives such as mixup-based diversity, anchor-based separation, or gradient-based disagreement could be compared against the original BEARS objective.

5. **RS Detection Score:** BEARS provides uncertainty at inference time. A thresholding mechanism to automatically flag high-RS inputs (as opposed to just reporting aggregate entropy) would make it deployment-ready.

---

## 9. File Structure Reference

```
bears/
├── XOR_MNIST/                    # Main experimental codebase
│   ├── main.py                   # Entry point for all MNIST-based experiments
│   ├── exp_best_args.py          # Pre-tuned hyperparameters per dataset/model
│   ├── experiments.py            # Experiment launchers (grid search configs)
│   ├── datasets/                 # Dataset loaders
│   │   ├── addmnist.py           # Addition-MNIST
│   │   ├── halfmnist.py          # Half-MNIST
│   │   ├── shortcutmnist.py      # Shortcut-MNIST
│   │   ├── restrictedmnist.py    # Restricted/Multi-op MNIST
│   │   ├── kandinsky.py          # Kandinsky Patterns
│   │   └── minikandinsky.py      # Mini Kandinsky (smaller version)
│   ├── models/                   # NeSy model implementations
│   │   ├── mnistdpl.py           # DPL on MNIST
│   │   ├── mnistsl.py            # SL on MNIST
│   │   ├── mnistltn.py           # LTN on MNIST
│   │   ├── mnistdplrec.py        # DPL + RECON
│   │   ├── mnistslrec.py         # SL + RECON
│   │   ├── mnistltnrec.py        # LTN + RECON
│   │   ├── mnistpcbmdpl.py       # DPL + PCBM
│   │   ├── kanddpl.py            # DPL on Kandinsky
│   │   └── ...
│   ├── utils/
│   │   ├── bayes.py              # BEARS ensemble, MC Dropout, Laplace
│   │   ├── test.py               # Evaluation pipeline (all 4 modes)
│   │   ├── train.py              # Training loop
│   │   ├── metrics.py            # Y-Acc, C-Acc, H(C), ECE
│   │   └── ...
│   ├── backbones/                # CNN encoders (ResNet, simple CNN, etc.)
│   ├── data/                     # Dataset files (MNIST auto-downloads here)
│   │   └── kand-3k.zip           # Kandinsky data (already present)
│   ├── dumps/                    # Output CSVs from evaluation runs
│   └── example/                  # XOR toy experiment (self-contained)
│       ├── xor_main.py           # XOR entry point
│       ├── dpl_models.py         # DPL for XOR
│       ├── sl_models.py          # SL for XOR
│       ├── ltn_models.py         # LTN for XOR
│       ├── dpl_train.py          # DPL training loop
│       ├── nesy_losses.py        # Shannon entropy loss
│       └── cfs/                  # Confusion matrix outputs
├── BDD_OIA/                      # Autonomous driving experiments (separate)
│   ├── main_bdd.py               # BDD-OIA entry point
│   └── ...
├── CITATION.cff                  # Paper citation
└── PROJECT_OVERVIEW.md           # This file
```

---

## 10. Current Status

- [x] Codebase cloned and understood
- [x] Paper read and summarized
- [x] All datasets identified and documented
- [ ] Phase 1: XOR environment check
- [ ] Phase 2: HalfMNIST baseline + BEARS
- [ ] Phase 3: AddMNIST baseline + BEARS
- [ ] Phase 4: Kandinsky baseline + BEARS
- [ ] Phase 5: Results consolidated into table
- [ ] Novelty direction selected and scoped
