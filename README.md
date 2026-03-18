<div align="center">

# 🌿 Carbon-Budgeted Evaluation of CNN Architectures on CIFAR Datasets

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![CodeCarbon](https://img.shields.io/badge/CodeCarbon-Emission%20Tracking-2D9B3A?style=for-the-badge&logo=leaflet&logoColor=white)](https://github.com/mlco2/codecarbon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange?style=for-the-badge)]()
[![Platform](https://img.shields.io/badge/Platform-Colab%20%7C%20Local-blue?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

> A framework for evaluating deep learning models under fixed CO₂ emission constraints, revisiting conventional accuracy-centric benchmarking through the lens of computational sustainability.

</div>

---

## 📖 Overview

This repository accompanies the paper:

> **"Carbon-Budgeted Evaluation of Convolutional Neural Network Architectures on CIFAR Datasets"**

Standard CNN benchmarking measures final accuracy after full training, which obscures a model's efficiency under real-world resource constraints. This work proposes a **carbon-budgeted evaluation paradigm**: given a predefined CO₂ emission budget (in grams), what is the maximum accuracy a model can achieve *before* that budget is exhausted?

The framework operates entirely via **post-hoc analysis of training logs** — no modifications to training procedures are required. Models are compared across three evaluation protocols: fixed-epoch training, convergence-based training, and carbon-budgeted evaluation.

---

## 🏆 Key Contributions

- ♻️ A **carbon-constrained evaluation framework** applicable to any CNN trained with per-epoch emission tracking.
- 📊 Systematic comparison of **six architectures** across three evaluation protocols on CIFAR-10 and CIFAR-100.
- 🔍 Empirical demonstration that final-accuracy rankings are **not preserved** under carbon constraints.
- ⚡ Identification of **early-learning efficiency** as a distinct and practically relevant performance metric.

---

## 🧠 Models Evaluated

| Model | Type | Characteristics |
|---|---|---|
| 🔵 LeNet-5 | Custom | Shallow baseline; minimal parameter count |
| 🟠 VGG-16 | Keras App | Deep, wide architecture; high emission profile |
| 🟢 ResNet-18 | Custom | Residual connections; strong early-epoch learning |
| 🔴 ResNet-50 | Keras App | Deeper residual variant; increased computational cost |
| 🟡 MobileNetV2 | Keras App | Depthwise separable convolutions; efficiency-oriented |
| 🟣 EfficientNet-B0 | Keras App | Compound-scaled; high accuracy but slower to ramp |

> All models are trained **from scratch** (`weights=None`) on 32×32 CIFAR inputs.

---

## 📦 Datasets

| Dataset | Classes | Image Size | Split |
|---|---|---|---|
| CIFAR-10 | 10 | 32 × 32 | 50,000 train / 10,000 test |
| CIFAR-100 | 100 | 32 × 32 | 50,000 train / 10,000 test |

Both datasets are used **as-is** without resizing, providing a controlled benchmarking environment that highlights architectural differences rather than data preprocessing effects.

---

## ⚙️ Methodology

### 📡 Carbon Tracking

Emissions are tracked using [CodeCarbon](https://github.com/mlco2/codecarbon), which estimates CO₂ equivalent (in grams) based on hardware power draw, runtime, and regional electricity carbon intensity:

```
E = P × t × I
```

| Symbol | Meaning |
|---|---|
| `E` | CO₂ emissions (g CO₂eq) |
| `P` | Hardware power draw (kW) |
| `t` | Training time per epoch (hours) |
| `I` | Regional grid carbon intensity (gCO₂/kWh) |

Emissions are logged cumulatively **after each epoch** and stored in `tracking/emissions.csv`.

---

### 🔬 Evaluation Protocols

**1️⃣ Fixed-Epoch Training**
All models trained for a uniform number of epochs. Final accuracy reported at the last epoch.

**2️⃣ Convergence-Based Training**
Training halted via early stopping when validation loss fails to improve for a patience window. Final accuracy at the early-stopping checkpoint is reported.

**3️⃣ Carbon-Budgeted Evaluation**
Post-hoc analysis of training logs under a hard emission constraint:
- Cumulative CO₂ tracked per epoch.
- Best validation accuracy within budget threshold returned.
- Budget thresholds: **1g, 5g, and 10g CO₂eq**

---

## 🧮 Algorithm: Carbon-Budgeted Evaluation

```
Input:   training_log (per-epoch accuracy and cumulative CO₂)
         carbon_budget B (grams CO₂eq)

Output:  best_accuracy within budget B

best_accuracy    ← 0
cumulative_carbon ← 0

for each epoch e in training_log:
    cumulative_carbon += emissions[e]
    if cumulative_carbon > B:
        break
    if accuracy[e] > best_accuracy:
        best_accuracy ← accuracy[e]

return best_accuracy
```

This procedure is **model-agnostic** and requires only a per-epoch log of accuracy and carbon emissions.

---

## 📈 Key Results

### CIFAR-10

- 🥇 **ResNet-18** achieves the best early-learning efficiency — reaches high accuracy within the 1g budget while emitting significantly less than deeper counterparts.
- 🥈 **MobileNetV2** performs competitively under moderate budgets (5g), benefiting from its parameter-efficient design.
- ❌ **VGG-16** and **EfficientNet-B0** require substantially larger budgets to reach peak accuracy, making them poor choices under emission constraints.
- ⬇️ **LeNet-5**, while very cheap to train, saturates at lower accuracy and does not scale favorably.

### CIFAR-100

- Class complexity amplifies efficiency differences. Architectures that appear similarly ranked under fixed-epoch training **diverge significantly** under budget-constrained evaluation.
- ResNet-18 retains its efficiency advantage; MobileNetV2 falls behind at stricter budgets.

> 💡 **Core insight:** Final-accuracy rankings do not reliably predict which model performs best under carbon constraints. **Early convergence behavior is the dominant factor.**

---

## 🌍 Why Carbon-Budgeted Evaluation Matters

Reporting final accuracy after exhaustive training does not reflect deployment reality. In practice:

- 💸 **Compute resources are finite.** Cloud ML workloads operate under cost and time budgets that translate directly to emission ceilings.
- 🎯 **Early performance is predictive.** Models that learn efficiently in early epochs are better candidates for interrupted or resource-constrained training pipelines.
- 🌱 **Sustainability requires measurable metrics.** Carbon-budgeted accuracy provides a concrete, reproducible axis on which to compare architectures — one that aligns model evaluation with environmental responsibility.

This framework reframes the question from *"which model is most accurate?"* to *"which model is most accurate **per unit of carbon emitted**?"*

---

## 🗂️ Repository Structure

```
.
├── 📓 Carbon_budgeting/
│   ├── cifar-10/
│   │   ├── models/                       # Modular per-model Python packages
│   │   │   ├── __init__.py               # get_model() + preprocess_data() registry
│   │   │   ├── lenet5.py
│   │   │   ├── vgg16.py
│   │   │   ├── resnet18.py
│   │   │   ├── resnet50.py
│   │   │   ├── mobilenetv2.py            # Includes MobileNetV2-specific preprocessing
│   │   │   └── efficientnetb0.py         # Includes EfficientNetB0-specific preprocessing
│   │   ├── experiment.ipynb              # Main notebook: import models, run, track carbon
│   │   ├── tracking/                     # Auto-created; stores emissions.csv
│   │   └── results/cifar10/              # Auto-created; stores per-model epoch CSVs
│   │
│   └── cifar-100/
│       ├── models/                       # Same structure, num_classes=100
│       │   ├── __init__.py
│       │   ├── lenet5.py  ·  vgg16.py  ·  resnet18.py
│       │   ├── resnet50.py  ·  mobilenetv2.py  ·  efficientnetb0.py
│       ├── run_experiment.ipynb          # Main notebook for CIFAR-100
│       ├── tracking/
│       └── results/cifar100/
│
├── 🧪 green-ai-cnn-carbon_CIFAR-10/
│   ├── models/                           # Script-based model definitions
│   ├── training/
│   │   ├── train_fixed_epochs.py
│   │   ├── train_convergence.py
│   │   └── callbacks.py
│   ├── tracking/
│   │   ├── carbon_tracker.py
│   │   └── emissions.csv
│   ├── results/
│   └── figures/
│
├── 🧪 green-ai-cnn-carbon_CIFAR-100/
│   ├── models/
│   ├── training/
│   ├── tracking/
│   └── results/
│
└── 📄 README.md
```

---

## 🔁 Reproducibility

| Aspect | Detail |
|---|---|
| 🔢 Trials | 3 independent runs, results averaged |
| 🖥️ Hardware | NVIDIA T4 GPU (Google Colab) |
| 🎛️ Hyperparameters | No architecture-specific tuning; uniform config across all models |
| 📊 Evaluation | Entirely post-hoc — deterministic from logged data, no randomness at evaluation |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/carbon-budgeted-cnn-evaluation.git
cd carbon-budgeted-cnn-evaluation

# Install core dependencies
pip install tensorflow codecarbon numpy pandas matplotlib
```

> **Google Colab:** `codecarbon` is auto-installed by the notebook on first run — no manual setup needed.

---

## 💻 Usage

### 🔬 Carbon-Budgeted Evaluation (Notebook)

Open the appropriate notebook and run all cells:

```
Carbon_budgeting/cifar-10/experiment.ipynb       ← CIFAR-10
Carbon_budgeting/cifar-100/run_experiment.ipynb  ← CIFAR-100
```

To switch models, change a **single line** in the *Model Selection* cell:

```python
# Options: 'lenet5' | 'vgg16' | 'resnet18' | 'resnet50' | 'mobilenetv2' | 'efficientnetb0'
MODEL_NAME = 'resnet18'
```

The notebook automatically:
- ✅ Installs `codecarbon` if missing
- ✅ Detects Google Colab and mounts Drive
- ✅ Applies the correct preprocessing per model (MobileNetV2 and EfficientNetB0 use their respective `preprocess_input` functions)
- ✅ Saves per-epoch results and carbon tracking logs

---

### 🏋️ Script-Based Training (Fixed-Epoch)

```bash
python green-ai-cnn-carbon_CIFAR-10/training/train_fixed_epochs.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100
```

### 🏋️ Script-Based Training (Convergence-Based)

```bash
python green-ai-cnn-carbon_CIFAR-10/training/train_convergence.py \
    --model mobilenetv2 \
    --dataset cifar10 \
    --patience 10
```

---

## 📐 Model Preprocessing Reference

| Model | Preprocessing Applied |
|---|---|
| LeNet-5 | None (input: `[0, 1]` float) |
| VGG-16 | None (input: `[0, 1]` float) |
| ResNet-18 | None (input: `[0, 1]` float) |
| ResNet-50 | None (input: `[0, 1]` float) |
| MobileNetV2 | `preprocess_input(x × 255)` → `[−1, 1]` |
| EfficientNet-B0 | `preprocess_input(x × 255)` → Keras internal scaling |

---

## 📝 Citation

If you use this framework, codebase, or findings in your research, please cite:

```bibtex
@article{bisaria2026carbonbudgeted,
  title   = {Carbon-Budgeted Evaluation of Convolutional Neural Network Architectures on CIFAR Datasets},
  author  = {Divye Bisaria and Ishan Gautam and Ankur Pandey},
  journal = {Under Review},
  year    = {2026}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

*For questions or issues, please open a GitHub Issue or contact the corresponding author.*

🌱 *Evaluating models through the lens of sustainability.*

</div>
