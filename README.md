# Carbon-Budgeted Evaluation of Convolutional Neural Network Architectures on CIFAR Datasets

> A framework for evaluating deep learning models under fixed CO₂ emission constraints, revisiting conventional accuracy-centric benchmarking through the lens of computational sustainability.

---

## Overview

This repository accompanies the paper:

**"Carbon-Budgeted Evaluation of Convolutional Neural Network Architectures on CIFAR Datasets"**

Full paper available at: [[link]](#)

Standard CNN benchmarking measures final accuracy after full training, which obscures a model's efficiency under real-world resource constraints. This work proposes a **carbon-budgeted evaluation paradigm**: given a predefined CO₂ emission budget (in grams), what is the maximum accuracy a model can achieve before that budget is exhausted?

The framework operates entirely via **post-hoc analysis of training logs** — no modifications to training procedures are required. Models are compared across three evaluation protocols: fixed-epoch training, convergence-based training, and carbon-budgeted evaluation.

---

## Key Contributions

- A carbon-constrained evaluation framework applicable to any CNN trained with per-epoch emission tracking.
- Systematic comparison of six architectures across three evaluation protocols on CIFAR-10 and CIFAR-100.
- Empirical demonstration that final-accuracy rankings are not preserved under carbon constraints.
- Identification of **early-learning efficiency** as a distinct and practically relevant performance metric.

---

## Models Evaluated

| Model           | Characteristics                                    |
|-----------------|----------------------------------------------------|
| LeNet-5         | Shallow baseline; minimal parameter count          |
| VGG-16          | Deep, wide architecture; high emission profile     |
| ResNet-18       | Residual connections; strong early-epoch learning  |
| ResNet-50       | Deeper residual variant; increased computational cost |
| MobileNetV2     | Depthwise separable convolutions; efficiency-oriented |
| EfficientNet-B0 | Compound-scaled; high accuracy but slower to ramp  |

---

## Datasets

| Dataset   | Classes | Image Size | Split            |
|-----------|---------|------------|------------------|
| CIFAR-10  | 10      | 32 × 32    | 50k train / 10k test |
| CIFAR-100 | 100     | 32 × 32    | 50k train / 10k test |

Both datasets are used as-is without resizing, providing a controlled benchmarking environment that highlights architectural differences rather than data preprocessing effects.

---

## Methodology

### Carbon Tracking

Emissions are tracked using [CodeCarbon](https://github.com/mlco2/codecarbon), which estimates CO₂ equivalent (in grams) based on hardware power draw, runtime, and regional electricity carbon intensity:

```
E = P × t × I
```

Where:
- `E` — CO₂ emissions (g CO₂eq)
- `P` — hardware power draw (kW)
- `t` — training time per epoch (hours)
- `I` — regional grid carbon intensity (gCO₂/kWh)

Emissions are logged cumulatively after each epoch and stored in `tracking/emissions.csv`.

---

### Evaluation Protocols

**1. Fixed-Epoch Training**  
All models trained for a uniform number of epochs. Final accuracy reported at the last epoch.

**2. Convergence-Based Training**  
Training halted via early stopping when validation loss fails to improve for a patience window. Final accuracy at the early-stopping checkpoint is reported.

**3. Carbon-Budgeted Evaluation**  
Post-hoc analysis of training logs under a hard emission constraint:
- Cumulative CO₂ is tracked per epoch.
- Evaluation proceeds until the cumulative emissions exceed the budget threshold.
- The best validation accuracy observed within the budget is returned.

Budget thresholds evaluated: **1g, 5g, and 10g CO₂eq**.

---

## Algorithm: Carbon-Budgeted Evaluation

```
Input:   training_log (per-epoch accuracy and cumulative CO₂),
         carbon_budget B (grams CO₂eq)

Output:  best_accuracy within budget B

best_accuracy ← 0
cumulative_carbon ← 0

for each epoch e in training_log:
    cumulative_carbon += emissions[e]
    if cumulative_carbon > B:
        break
    if accuracy[e] > best_accuracy:
        best_accuracy ← accuracy[e]

return best_accuracy
```

This procedure is model-agnostic and requires only a per-epoch log of accuracy and carbon emissions.

---

## Key Results

### CIFAR-10

- **ResNet-18** achieves the best early-learning efficiency: it reaches high accuracy within the 1g budget while emitting significantly less than deeper counterparts.
- **MobileNetV2** performs competitively under moderate budgets (5g), benefiting from its parameter-efficient design.
- **VGG-16** and **EfficientNet-B0** require substantially larger budgets to reach peak accuracy, making them poor choices when emissions are constrained.
- **LeNet-5**, while very cheap to train, saturates at lower accuracy and does not scale favorably.

### CIFAR-100

- Class complexity amplifies efficiency differences. Architectures that appear similarly ranked under fixed-epoch training diverge significantly under budget-constrained evaluation.
- ResNet-18 retains its efficiency advantage; MobileNetV2 falls behind at stricter budgets where it cannot achieve sufficient accuracy early in training.

**Core insight**: Final-accuracy rankings do not reliably predict which model performs best under carbon constraints. Early convergence behavior is the dominant factor.

---

## Visualizations

The repository includes the following figures (generated in `figures/`):

- **Accuracy vs. Cumulative CO₂ curves** — per-model training trajectories plotted against emissions.
- **Pareto frontiers** — accuracy-vs-carbon trade-off under fixed-epoch and convergence-based protocols.
- **Carbon-budget bar plots** — best achieved accuracy per model at 1g, 5g, and 10g budget thresholds.

---

## Why Carbon-Budgeted Evaluation Matters

Reporting final accuracy after exhaustive training does not reflect deployment reality. In practice:

- **Compute resources are finite.** Cloud ML workloads operate under cost and time budgets that translate directly to emission ceilings.
- **Early performance is predictive.** Models that learn efficiently in early epochs are better candidates for interrupted or resource-constrained training pipelines.
- **Sustainability requires measurable metrics.** Carbon-budgeted accuracy provides a concrete, reproducible axis on which to compare architectures — one that aligns model evaluation with environmental responsibility.

This framework does not penalize accuracy; it reframes the question from *"which model is most accurate?"* to *"which model is most accurate per unit of carbon emitted?"*

---

## Repository Structure

```
.
├── green-ai-cnn-carbon_CIFAR-10/
│   ├── models/                  # Model definitions (LeNet, VGG16, ResNet, MobileNetV2, EfficientNet)
│   ├── training/
│   │   ├── train_fixed_epochs.py
│   │   ├── train_convergence.py
│   │   └── callbacks.py
│   ├── tracking/
│   │   ├── carbon_tracker.py    # CodeCarbon wrapper
│   │   └── emissions.csv        # Per-run emission logs
│   ├── results/
│   │   ├── logs/                # Per-epoch accuracy logs
│   │   ├── raw/                 # Raw result CSVs
│   │   └── processed/           # Aggregated results
│   ├── experiments/             # Experiment notebooks
│   └── figures/                 # Generated plots
│
├── green-ai-cnn-carbon_CIFAR-100/
│   ├── models/
│   ├── training/
│   ├── tracking/
│   └── results/
│
└── Carbon_budgeting/
    ├── cifar-10/
    │   └── experiment.ipynb     # Carbon-budgeted evaluation notebook (CIFAR-10)
    └── cifar-100/
        └── experiment.ipynb     # Carbon-budgeted evaluation notebook (CIFAR-100)
```

---

## Reproducibility

- Each experiment was run for **3 independent trials** with results averaged.
- All experiments were conducted on a single **NVIDIA T4 GPU** (Google Colab environment).
- No architecture-specific hyperparameter tuning was applied; all models use identical training configurations.
- Carbon-budgeted evaluation is entirely post-hoc: results are derived deterministically from logged data, with no randomness introduced at evaluation time.

---

## Installation

```bash
git clone https://github.com/<your-username>/carbon-budgeted-cnn-evaluation.git
cd carbon-budgeted-cnn-evaluation
pip install -r requirements.txt
```

**Core dependencies:** PyTorch, torchvision, CodeCarbon, NumPy, pandas, matplotlib.

---

## Usage

### Training (Fixed-Epoch)

```bash
python green-ai-cnn-carbon_CIFAR-10/training/train_fixed_epochs.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100
```

### Training (Convergence-Based)

```bash
python green-ai-cnn-carbon_CIFAR-10/training/train_convergence.py \
    --model mobilenetv2 \
    --dataset cifar10 \
    --patience 10
```

### Carbon-Budgeted Evaluation

Open and run the appropriate notebook:

```
Carbon_budgeting/cifar-10/experiment.ipynb
Carbon_budgeting/cifar-100/experiment.ipynb
```

These notebooks load the emission logs from `tracking/emissions.csv` and the accuracy logs from `results/logs/`, then apply the budgeted evaluation algorithm across all budget thresholds.

---

## Citation

If you use this framework, codebase, or findings in your research, please cite:

```bibtex
@article{yourname2026carbonbudgeted,
  title     = {Carbon-Budgeted Evaluation of Convolutional Neural Network Architectures on CIFAR Datasets},
  author    = {[Author(s)]},
  journal   = {[Journal Name]},
  year      = {2026},
  note      = {[DOI or arXiv link]}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

*For questions or issues, please open a GitHub issue or contact the corresponding author.*
