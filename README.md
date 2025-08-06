# FitNets Feature-Based Distillation (PyTorch)

This project implements FitNets-style hint-based knowledge distillation in PyTorch and tracks GPU energy usage.

## 🔍 Key Components

- **Teacher**: Deeper CNN trained on CIFAR-10
- **Student**: Thinner CNN trained with:
  - CE loss
  - KL divergence (soft logits)
  - Hint loss (MSE on intermediate features)
- **Monitoring**: GPU power tracking using pynvml

## 📦 Usage

### 1. Train with Distillation + Power Logging

```bash
python run_distill.py
```

### 2. Inference + Power Logging

```bash
python run_inference.py --model teacher
python run_inference.py --model student
```

## 📊 Output

- `logs/distill_power.csv`: Power logs during training
- `logs/infer_student.csv`: Inference energy of student
- `logs/infer_teacher.csv`: Inference energy of teacher


## 🧪 Requirements

```bash
pip install requirements.txt
```
