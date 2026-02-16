🌱 Carbon-Budgeted Evaluation of CNN Architectures on CIFAR

Rethinking CNN evaluation under environmental constraints.

This repository accompanies the research paper:

"Carbon-Budgeted Evaluation of Convolutional Neural Network Architectures on CIFAR Datasets"
Dr. Ankur Pandey, Divye Bisaria*, Ishan Gautam
Manipal University Jaipur, India

📌 Corresponding Author: divyebisaria4106@gmail.com

📖 Overview

Deep learning models are traditionally evaluated using final accuracy after fixed epochs or convergence. However, this ignores carbon emissions and environmental cost during training.

This project introduces a Carbon-Budgeted Evaluation Framework that:

Treats carbon emissions as a primary constraint

Analyzes training trajectories post-hoc

Determines maximum achievable accuracy within a fixed CO₂ budget

Enables fair architectural comparison under sustainability constraints

Instead of asking:

"Which model achieves the highest accuracy?"

We ask:

"Which model achieves the best accuracy within a given carbon budget?"

🧠 Architectures Evaluated

The study covers CNN models across different design generations:

🔹 LeNet-5
4

Early CNN architecture

Simple convolution + pooling design

Baseline carbon-efficient model

🔹 VGG-16
4

Deep sequential architecture

High accuracy

Carbon-inefficient under constrained budgets

🔹 ResNet-18 / ResNet-50
4

Introduced residual (skip) connections

Strong early learning efficiency

Dominates under strict carbon budgets

🔹 MobileNetV2
4

Lightweight architecture

Designed for mobile efficiency

Competitive under moderate budgets

🔹 EfficientNet-B0
4

Compound scaling (depth, width, resolution)

Extremely low emissions in some settings

Delayed early learning behavior

📊 Datasets

CIFAR-10 (10 classes)

CIFAR-100 (100 classes)

32×32 RGB images

No data augmentation (to isolate architectural effects)

⚙️ Experimental Setup
Component	Configuration
Optimizer	Adam
Loss	Categorical Cross-Entropy
LR Schedule	Cosine Decay
Batch Size	64
Fixed Epochs	50
Max Epochs	100
Early Stopping	Patience = 10
Runs per Model	3
Hardware	NVIDIA T4 GPU (Google Colab)
Carbon Tracking	CodeCarbon
🌍 Carbon Emission Estimation

Carbon emissions are estimated as:

𝐸
𝑐
𝑎
𝑟
𝑏
𝑜
𝑛
=
𝑃
×
𝑡
×
𝐼
E
carbon
	​

=P×t×I

Where:

P = Average power (kW)

t = Training time (hours)

I = Carbon intensity (kg CO₂e/kWh)

Tracking is performed using CodeCarbon during training.

🧮 Carbon-Budgeted Training Algorithm
Input: Model M, Dataset D, Max epochs E, Carbon budget B
Initialize model parameters
C_total = 0

for epoch in 1 to E:
    Train model for one epoch
    Measure C_epoch
    C_total += C_epoch
    
    if C_total ≥ B:
        Stop training

Return trained parameters

📌 Training dynamics are NOT modified.
📌 Carbon is used strictly as a stopping condition.
📌 Evaluation is architecture-independent and post-hoc.

📈 Key Findings
🔥 1. Residual Networks Are Carbon-Efficient

ResNet-18 consistently dominates under strict budgets

Early learning efficiency matters more than final accuracy

⚠️ 2. VGG Is Carbon-Inefficient

Strong convergence accuracy

Poor early learning → performs badly under carbon limits

📉 3. Dataset Complexity Matters

CIFAR-100 magnifies efficiency differences

Early learning becomes more critical

🌱 4. Final Accuracy Can Be Misleading

Traditional benchmarks hide:

Early learning behavior

Carbon cost accumulation

Efficiency trade-offs

📌 Why This Matters

This framework helps:

Sustainable AI research

Industry with carbon/time constraints

Carbon-aware model selection

Resource-constrained deployment environments

It shifts evaluation from:

Performance-only

to

Performance under environmental constraints

🚧 Limitations

Hardware-specific (NVIDIA T4 GPU)

Training-time emissions only

No inference carbon accounting

Carbon estimation approximated via tracking tools

🔮 Future Work

Larger datasets (ImageNet-scale)

Multi-objective constraints (carbon + time + cost)

Inference-time carbon analysis

Hardware-aware architectural comparisons

📜 Citation

If you use this work, please cite:

@article{bisaria2026carbon,
  title={Carbon-Budgeted Evaluation of CNN Architectures on CIFAR Datasets},
  author={Pandey, Ankur and Bisaria, Divye and Gautam, Ishan},
  journal={Under Review},
  year={2026}
}
👨‍💻 Author

Divye Bisaria
B.Tech CSE
Manipal University Jaipur
📧 divyebisaria4106@gmail.com

⭐ Contributing

Contributions are welcome!

If you’d like to:

Extend to new architectures

Add inference carbon evaluation

Apply to large-scale datasets

Improve reproducibility

Please open an issue or pull request.
