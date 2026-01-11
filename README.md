# Kalman-Style Smoothing for Robust Learning  
### Final Project for *Seminar Geometry and Topology in Deep Learning*

---

## 📌 Overview

This project explores **Kalman-style smoothing** as a general and robust strategy for learning under noisy statistics.  
Rather than treating Kalman filtering as a strict state-space estimator, we adopt it as a **geometric and statistical smoothing principle**, applicable across different learning paradigms.

The project consists of **two complementary parts**:

1. **Normalization and Optimization Geometry in Deep Networks**  
2. **Kalman-Based Reward Normalization in Reinforcement Learning**

Together, these parts demonstrate how temporal smoothing of noisy signals can significantly stabilize training dynamics, improve robustness, and provide interpretable geometric structure in learning trajectories.

---
## 🔧 Environment Setup

For part 1, the repository was tested with **Python ≥ 3.8** and **PyTorch ≥ 1.10**.  
The codebase is lightweight and does not rely on any non-standard dependencies.

We recommend using a virtual environment.


### Part 1: Conda (Recommended)

```bash
conda create -n kalman_norm python=3.9 -y
conda activate kalman_norm
```

Install required packages:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

### Part 2: Google Colab

Note on a Weird Runtime Issue

⚠️ Important:
If you encounter an error when running the code below, please rerun the two cells above first, and then rerun the current cell.

This issue is admittedly a bit strange. From our experience, it seems to be caused by an inconsistency between the Gym and NumPy environments, possibly related to version mismatches or how the runtime handles package states.

If rerunning the two cells once does not fix the problem, try rerunning them multiple times (yes, really). In some cases, the error disappears after a few retries. Unfortunately, we do not yet have a clean or principled explanation for why this happens.





---

## 🧩 Part I: Kalman-Style Smoothing for Normalization  
### (Image Classification with ResNet on CIFAR)


### 🚀 Quick Start: CIFAR-10 Training (Recommended)

This project provides a **single, minimal entry point** to reproduce the core experimental results used in the final presentation for the *Seminar on Geometry and Topology in Deep Learning*.

All CIFAR-10 experiments are run via:

```bash
python examples/cifar10_train.py \
  --norm_type gkn \
  --num_groups 4 \
  --p_rate 0.1
```

### Motivation

Normalization layers implicitly define a **coordinate system and local geometry** in feature space.  
However, commonly used methods such as Batch Normalization and Group Normalization rely on instantaneous or weakly smoothed statistics, which can be noisy under small batch sizes or non-ideal training conditions.

We propose **Kalman-Style Smoothing for Normalization**, instantiated as **Group Kalman-Inspired Normalization (GKN)**, which:
- Computes GroupNorm-style per-sample statistics
- Applies **temporal smoothing** to group-level moments
- Reduces variance in statistical estimation without sacrificing convergence or accuracy

### Key Idea

Training dynamics are viewed as a **trajectory on a loss-induced manifold**.  
Kalman-style smoothing acts as a low-pass filter along the *time dimension*, leading to:
- Smoother optimization paths
- Reduced curvature and noise in training trajectories
- More stable generalization behavior

### Experiments

- Dataset: CIFAR-10 / CIFAR-100  
- Architecture: ResNet variants  
- Baselines: BatchNorm (BN), GroupNorm (GN)  
- Metrics:
  - Train / Test Loss and Accuracy
  - Geometric trajectory analysis in loss and accuracy space
  - Discrete speed, curvature (turning angle), and path length
 
### 📊 Results: Image Classification Results (Test Accuracy)

| Dataset    | Normalization | Test Accuracy (%) |
|------------|---------------|-------------------|
| CIFAR-10   | GN            | 88.12             |
| CIFAR-10   | GKN (Ours)    | **88.47**         |
| CIFAR-100  | GN            | 59.63             |
| CIFAR-100  | GKN (Ours)    | **62.36**         |


### Geometric Analysis

We visualize training as curves in:
- **Loss space**: (L_train, L_test)
- **Accuracy space**: (Acc_train, Acc_test)

Compared to GN, GKN consistently exhibits:
- Lower trajectory curvature
- Reduced high-frequency oscillations
- More stable coupling between training and generalization

These observations support a geometric interpretation of normalization as a **metric regularizer on training dynamics**. Please find the plots in final_project.ipynb

**Geometric interpretation (accuracy space).**  
We view training as a discrete trajectory γ(t) = (train_acc(t), test_acc(t)) in a 2D generalization space.  
Compared to GN, GKN yields a substantially shorter trajectory (total arclength 132.7 vs 155.9) and smaller average step length (1.34 vs 1.58), indicating fewer epoch-to-epoch oscillations and less “wandering” in the train–test accuracy coupling.  

Although turning-angle statistics are comparable in this projection, the reduced path length and step magnitude strongly support the claim that Kalman-style smoothing stabilizes training dynamics and improves robustness in the generalization trajectory.


---

## 🧠 Part II: Kalman-Based Reward Normalization in Reinforcement Learning  
### (Policy Gradient Methods)

### Background and Motivation

Reinforcement learning often suffers from **high-variance and non-stationary reward signals**, which severely degrade training stability and sample efficiency—especially in policy gradient methods.

In this part, we explore **Kalman-based reward normalization**, replacing conventional mean–std normalization with a lightweight **1D Kalman filter** applied directly to reward streams.

### Method

- Rewards are modeled as noisy observations of an underlying latent signal
- A 1D Kalman filter provides adaptive, online normalization
- The approach is plug-and-play, computationally lightweight, and compatible with standard RL pipelines

### Algorithms and Environments

- Algorithms: Actor-Critic (AC), PPO  
- Environments: CartPole, LunarLander

### Results

Kalman-based reward normalization consistently provides:
- Faster early-stage convergence
- Reduced variance in learning curves
- Improved stability across random seeds

### Academic Context

This reinforcement learning work was previously submitted to the **ICML New in ML Workshop**, an inclusive venue aimed at encouraging exploratory machine learning research by early-stage researchers.

Although exploratory in nature, subsequent feedback and industry interest suggest that **Kalman-based reward normalization is a promising direction worthy of further investigation**.  
We therefore include this work as the second part of the present project.

---

## 🧭 Unifying Perspective

Across both parts, this project demonstrates that:

> **Kalman-style smoothing provides a unifying geometric and statistical framework for stabilizing learning under noise.**

- In supervised learning, it smooths normalization statistics and shapes optimization geometry.
- In reinforcement learning, it smooths reward signals and stabilizes policy updates.

---

## 🔮 Future Directions

- Learnable Kalman parameters (Q, R)
- Uncertainty-aware normalization layers
- State-dependent reward filtering
- Extensions to continuous control and offline RL
- Connections to information geometry and dynamical systems

---

## 📜 License

MIT License

## 📖 Citation

This project is closely related to our prior work on Kalman-based optimization methods,  
which was presented in the *Seminar on Geometry and Topology in Deep Learning*.

```bibtex
@inproceedings{xiakoala++,
  title     = {KOALA++: Efficient Kalman-Based Optimization with Gradient-Covariance Products},
  author    = {Xia, Zixuan and Davtyan, Aram and Favaro, Paolo},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
}


