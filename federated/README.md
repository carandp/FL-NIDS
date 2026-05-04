# Federated Learning — Quick Start Guide

> **Prerequisites:** `uv` installed and the dataset available at `datasets/fed_clients/`. The fed_clients folder is generated with data_analysis/data.ipynb.

---

## 1. Change kernel and install dependencies (if needed)

```bash
cd federated/
uv sync
source .venv/bin/activate
```

---

## 2. Setup

```bash
uv run python gen_config.py
```

---

## 3. Set up and start the POC environment

The first time (or after deleting `poc_workspace/`):

```bash
NVFLARE_POC_WORKSPACE=/home/<user>/FL-NIDS/federated/poc_workspace uv run nvflare poc prepare -i ./project.yml
```

Link job directory

```bash
NVFLARE_POC_WORKSPACE=/home/<user>/FL-NIDS/federated/poc_workspace uv run nvflare poc prepare-jobs-dir -j ./jobs
```

Then start the server, client, and admin console together:

```bash
uv run nvflare poc start
```

You will see a `>` prompt — this is the NVFlare admin console.

---

## 4. Submit the job

At the `>` prompt:

```bash
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

Stop the POC environment:

```bash
shutdown all
```

---

## 6. Evaluate the model

```bash
uv run python gen_eval_federated.py \
    --model jobs/<job_id>/workspace/app_server/FL_global_model.pt
```

The script derives an anomaly threshold from the validation set and reports:

```
Test macro F1-score : 0.XXXX
Test PR-AUC         : 0.XXXX
Prediction time     : X.XXXX s
```

```bash
uv run python gen_eval_clients.py
```

Will evaluate the clients of the last runned job.

## 7. Extras

To generate metrics_plots:

```bash
uv run python gen_metrics_graphs.py
```

To generate tsne_plots:

```bash
uv run python gen_tSNE_clients.py
``` 

or

```bash
uv run python gen_tSNE_server.py \
        --job <job_id>
```

To generate diagnostics:

```bash
uv run python gen_diagnostics_plots.py
```

```bash
uv run python gen_diagnostic_plot_global.py \
    --job <job_id> \
    --hist_min 0.0 \
    --hist_max 0.002
```

