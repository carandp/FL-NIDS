# Data Analysis — Quick Start Guide
Analyze and explore the network flow IDS dataset with Jupyter notebooks.

> **Prerequisites:** `uv` installed and the dataset available at `datasets/NF-CSE-CIC-IDS2018-v3/`. See the root README for setup details.

---

## 1. Install Dependencies

```bash
uv sync
```

## 2. Start Kernel

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project
```

## 3. Run the Analysis

1. Open `data.ipynb` in Jupyter Lab
2. Select the `project` kernel from the kernel selector
3. Execute the cells sequentially to run the analysis
