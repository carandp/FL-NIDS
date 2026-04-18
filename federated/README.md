# Federated Learning — Quick Start Guide

> **Prerequisites:** `uv` installed and the dataset available at `datasets/NF-CSE-CIC-IDS2018-v3/`. See the root README for setup details.

---

## 1. Change Kernel (if needed)

```bash
source .venv/bin/activate
```

---

## 2. Install dependencies and setup

```bash
cd federated/
uv sync && uv run python gen_config.py
```

---

## 3. Set up and start the POC environment

The first time (or after deleting `poc_workspace/`):

```bash
uv run nvflare poc prepare -i ./project.yml -d .
```

Link job directory

```bash
uv run nvflare poc prepare-jobs-dir -j ./jobs
```

Then start the server, client, and admin console together:

```bash
uv run nvflare poc start
```

You will see a `>` prompt — this is the NVFlare admin console.

---

## 4. Submit the job

At the `>` prompt:

```
submit_job nids_fedavg
```

The console prints the assigned job ID, e.g.:

```
Submitted job: c561bcf7-d45b-4cb5-9cc0-89b9ac8efed7
```

Training progress is printed to the same terminal. Each round logs the local epoch losses from every client.

---

## 5. Download the trained model

Once all rounds complete (you will see `FINISHED:COMPLETED` in the logs), download the job output at the `>` prompt:

```
download_job <job_id>
```

This extracts the workspace to:

```
jobs/<job_id>/workspace/app_server/FL_global_model.pt
```

Type `bye` to exit the admin console.

Stop the POC environment:

```bash
uv run nvflare poc stop
```

---

## 6. Evaluate the model

```bash
uv run python eval_federated.py \
    --model jobs/<job_id>/workspace/app_server/FL_global_model.pt \
    --dataset NF-CSE-CIC-IDS2018-v3
```

The script derives an anomaly threshold from the validation set and reports:

```
Test macro F1-score : 0.XXXX
Test PR-AUC         : 0.XXXX
Prediction time     : X.XXXX s
Peak GPU memory     : XXX.XX MB
```

### Optional flags

| Flag | Description |
|---|---|
| `--threshold_method unsupervised` | Use MAD-based threshold instead of best-F1 search |
| `--save_curve` | Save precision-recall curve to `curves/` as `.npz` |
| `--fraction 0.2` | Default — reuses the cached processed dataset from training |
