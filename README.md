# FL-NIDS
This repository evaluates the trade-offs between privacy, performance, and computational cost in Federated Network Intrusion Detection Systems (FL-NIDS) by layering Secure Aggregation and Differential Privacy on a GNN-based model using NVIDIA FLARE.

---

## 1. Project Structure

The project is organized into three main phases, each corresponding to a specific implementation milestone.

### 1.1 centralized/

This folder contains the Centralized Benchmark.

- **Goal.** Establish the maximum possible accuracy and F1-score using a standard local training environment.  
- **Dataset.** Utilizes NetFlow V3 ([NF-CSE-CIC-IDS2018-v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)).  
- **Model.** Self-supervised Graph Neural Network ([GraphIDS](https://github.com/lorenzo9uerra/GraphIDS)).

### 1.2 federated/

This folder contains the transition to Federated Learning using NVFlare.

- **Goal.** Benchmark the performance of a distributed model without extra privacy layers.  
- **Architecture.** Implements a controller-client integration where the FL controller assigns training tasks to multiple clients.

### 1.3 federated_hard/

This folder contains the Hardened Federated Model.

- **Secure Aggregation (SA).** Cryptographic protection ensuring the server only sees combined updates, mitigating server-side data leakage.  
- **Differential Privacy (DP).** Implementation of controlled random noise to protect against membership inference attacks.  
- **Parametric Study.** Contains the 25 simulations varying the noise multiplier (σ) to generate accuracy-privacy trade-off curves.

---

## 2. Getting Started

### 2.1 Prerequisites

- Linux / [WSL2](https://documentation.ubuntu.com/wsl/stable/howto/install-ubuntu-wsl2/)
- [uv](https://docs.astral.sh/uv/) (for project management)
- NetFlow V3 Dataset. You will need to download the University of Queensland feature-integrated [NF-CSE-CIC-IDS2018-v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) dataset.

### 2.2 Hardware

The hardware used for training and testing the models is the following.

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 5800x3D (8 cores) |
| GPU | NVIDIA RTX 4090 (24GB) |
| Memory | DDR4 RAM (32GB) |

### 2.3 Next Step: WIP 🚧 

---

## 9. License

All original components of this repository are licensed under the [Apache License 2.0](https://github.com/carandp/FL-NIDS/blob/main/LICENSE). Third-party components are used in compliance with their respective licenses.