# Adding Differential Privacy to an Existing NVFlare Project

This guide walks you through integrating the four DP files into a working
federated learning project. No changes to your model, trainer, or server
code are required — the filter slots in purely at the config level.

---

## Files produced and what each one does

| File | Role | Where it lives |
|---|---|---|
| `rdp_accountant.py` | Maths — tracks RDP privacy budget | same folder as your custom components |
| `budget_tracker.py` | Persistence — wraps accountant, writes JSON per round | same folder |
| `dp_gaussian_filter.py` | NVFlare `Filter` — clips + noises weight updates | same folder |
| `dp_noise_calibrator.py` | Offline tool — finds the right `noise_multiplier` | run once locally, not deployed |
| `config_fed_client.json` | Reference snippet — shows what to add to your client config | reference only, do not replace your config |

The dependency chain is linear and contained:

```
dp_gaussian_filter.py
    └── budget_tracker.py
            └── rdp_accountant.py
```

`dp_noise_calibrator.py` is standalone and only used before deployment.

---

## Step 0 — Run the noise calibrator first (do this before touching the project)

The calibrator tells you which `noise_multiplier` to use given your exact
dataset sizes, batch size, number of rounds, and privacy target.

```bash
# from the folder where you downloaded the files
python dp_noise_calibrator.py
```

It will print a table like this:

```
  σ=0.80  client_1    0.9301ε  client_2    0.8112ε  client_3    7.7281ε  ✓
  σ=1.10  client_1    0.4598ε  client_2    0.3844ε  client_3    4.0619ε  ✓
```

RESULTS OF RUNNING IT:

(federated-dp) carandp@DESKTOP-2TD7K2T:~/FL-NIDS/federated_dp$ uv run python materials/dp_noise_calibrator.py
     σ    client_1    client_2    client_3   all_ok
---------------------------------------------------
  0.50     4.1797ε     3.6239ε    41.4177ε        ✗
  0.55     2.8569ε     2.5638ε    25.4650ε        ✗
  0.60     2.1512ε     1.7924ε    18.9338ε        ✗
  0.65     1.7246ε     1.3985ε    14.2733ε        ✗
  0.70     1.3169ε     1.1856ε    11.0377ε        ✗
  0.75     1.1236ε     0.9430ε     9.2711ε        ✓

✓  Recommended noise_multiplier = 0.75
   Privacy budgets after 100 rounds (δ=1e-05):
       client_1:  ε = 1.1236
       client_2:  ε = 0.9430
       client_3:  ε = 9.2711

Pick the smallest σ where all clients show `✓` against your target ε.
The default target is `ε=10, δ=1e-5` over 100 rounds — edit the bottom
of `dp_noise_calibrator.py` if your setup differs.

Your three clients at `batch_size=32, local_epochs=2, T=100` already
satisfy `ε ≤ 10` even at `σ=0.8`, so `σ=1.1` gives you comfortable
headroom (worst-case client 3: ε ≈ 4.1).

Also run a few dry-run training rounds **without DP** and log the
`||Δw||` (L2 norm of weight updates). Set `clip_norm` at roughly the
70th percentile of what you observe. A `clip_norm` that is too small
relative to true update norms destroys model quality before noise even enters.

---

## Step 1 — Copy the three runtime files into your project

Your NVFlare project has a `custom/` folder (or whatever you have named
your component directory — the one listed in `PYTHONPATH` for clients).
Copy the three runtime files there:

```
jobs/nids_fedavg/app
├── custom/
│   ├── your_learner.py          ← already exists
│   ├── rdp_accountant.py        ← copy here
│   ├── budget_tracker.py        ← copy here
│   └── dp_gaussian_filter.py    ← copy here
├── config/
│   ├── config_fed_client.json   ← you will edit this
│   └── config_fed_server.json   ← leave untouched
```

All three files use only `numpy` and the Python standard library
(plus `nvflare` for the filter). No additional `pip install` is needed
beyond what your project already uses.

---

## Step 2 — Edit `config_fed_client.json`

You need to make two additions to each client's config: register the
filter as a component, then wire it into `task_result_filters`.

IMPORTANT: in this proyect there is a `config_fed_client.template.json` you must edit this one
cause then the real config is generated with gen_config.py! It doesnt change that much
it just adds user to a path and thats all.

### 2a. Add the component

Inside the `"components"` array, add this block alongside your existing
components (your learner, aggregator, etc.):

```json
{
  "id": "dp_gaussian_filter",
  "path": "dp_gaussian_filter.DPGaussianFilter",
  "args": {
    "dataset_size":     770415,
    "batch_size":       32,
    "local_epochs":     2,
    "clip_norm":        1.0,
    "noise_multiplier": 1.10,
    "target_epsilon":   10.0,
    "target_delta":     1e-5,
    "budget_dir":       "/tmp/nvflare_dp_budget"
  }
}
```

**Critical:** `dataset_size` must be set to the number of training rows
on that specific client. The three values for your project are:

| Client | `dataset_size` |
|---|---|
| client_1 | 770415 |
| client_2 | 1394658 |
| client_3 | 10960 |

Everything else (`clip_norm`, `noise_multiplier`, `target_epsilon`,
`target_delta`) should be identical across all three clients.

### 2b. Wire the filter into `task_result_filters`

Find the `"task_result_filters"` key in the config. If it does not exist,
add it. Insert the filter for the `"train"` task:

```json
"task_result_filters": [
  {
    "tasks": ["train"],
    "filters": ["dp_gaussian_filter"]
  }
]
```

If `task_result_filters` already has entries, just add `"dp_gaussian_filter"`
to the `"filters"` list of the `"train"` task. The DP filter should be
**last** in the list so it runs after any other result processing you have.

### What the final diff looks like

```jsonc
// config_fed_client.json (client_1 example)
{
  "components": [
    { /* your existing learner component */ },

    // ← ADD THIS BLOCK
    {
      "id": "dp_gaussian_filter",
      "path": "dp_gaussian_filter.DPGaussianFilter",
      "args": {
        "dataset_size":     770415,   // client-specific
        "batch_size":       32,
        "local_epochs":     2,
        "clip_norm":        1.0,
        "noise_multiplier": 1.10,
        "target_epsilon":   10.0,
        "target_delta":     1e-5,
        "budget_dir":       "/tmp/nvflare_dp_budget"
      }
    }
    // ← END
  ],

  // ← ADD OR EXTEND THIS SECTION
  "task_result_filters": [
    {
      "tasks": ["train"],
      "filters": ["dp_gaussian_filter"]
    }
  ]
}
```

---

## Step 3 — Ensure the budget directory is writable

The filter writes one JSON file per client under `budget_dir`.
The default is `/tmp/nvflare_dp_budget`, which works for local simulation.

For a real multi-machine deployment, set `budget_dir` to a path that
exists and is writable on each client machine, for example:

```json
"budget_dir": "/home/nvflare/dp_budget"
```

The directory is created automatically if it does not exist, as long as
the parent path is writable.

Specifically for this part of logging, there is a already working implementation in nids_trainer.py that 
logs macro-F1 and PR-AUC, so you can guide yourself with that!

```python
# Lines 249...
self.metrics_history.append({
            "round": current_round,
            "train_loss": total_train_loss,
            "val_loss": val_loss_local,
            "val_pr_auc": val_pr_auc_local,
            "val_macro_f1": val_f1_local,
        })
```

---

## Step 4 — Server config (no changes required)

The DP filter operates entirely on the client side. The server's
`config_fed_server.json` does not need any modification.

The only server-side consideration is how you handle the rare case where
a client exhausts its privacy budget and starts sending empty
`WEIGHT_DIFF` updates. Standard `FedAvg` ignores empty contributions
naturally, but if your aggregator has a strict minimum-clients check, make
sure it tolerates receiving fewer non-empty updates than expected in late
rounds.

---

## Step 5 — Verify the integration before full training

Run a short smoke test with `num_rounds=2` and confirm you see DP log
lines in the client output:

```
[DP/client_1] round=1  ε=0.0042 / 10.0  δ=1e-05  steps=48151  q=0.000042
[DP/client_2] round=1  ε=0.0024 / 10.0  δ=1e-05  steps=87166  q=0.000023
[DP/client_3] round=1  ε=0.0323 / 10.0  δ=1e-05  steps=685    q=0.002920
```

Also check that budget JSON files appeared in `budget_dir`:

```bash
ls /tmp/nvflare_dp_budget/
# client_1_budget.json  client_2_budget.json  client_3_budget.json
```

Inspect one to confirm it looks sane:

```bash
cat /tmp/nvflare_dp_budget/client_3_budget.json
```

```json
{
  "client_id": "client_3",
  "round_num": 2,
  "current_epsilon": 0.064,
  "target_epsilon": 10.0,
  "budget_exhausted": false,
  "history": [
    {"round": 1, "epsilon": 0.032},
    {"round": 2, "epsilon": 0.064}
  ],
  ...
}
```

---

## Step 6 — Full training run

Start training normally. No other change is needed. Budget JSON files are
updated after every round and survive crashes — if a run is interrupted and
resumed, the accountant picks up where it left off.

Monitor the `current_epsilon` in the JSON files (or in the client logs)
as training progresses. With `σ=1.1` and your dataset sizes, expected
final ε values after 100 rounds are approximately:

| Client | Expected final ε |
|---|---|
| client_1 | ≈ 0.46 |
| client_2 | ≈ 0.38 |
| client_3 | ≈ 4.06  ← binding client |

---

## Parameter reference

| Parameter | What it controls | How to choose |
|---|---|---|
| `clip_norm` | L2 bound on weight update norm | measure median `‖Δw‖` from a no-DP run; set to 70th–80th percentile |
| `noise_multiplier` | ratio σ/C; larger = more noise, better privacy | use `dp_noise_calibrator.py` output |
| `target_epsilon` | maximum allowed ε before the client stops sending updates | 1–10 is typical; lower = stronger privacy |
| `target_delta` | failure probability δ in (ε, δ)-DP | set to `1 / (10 × dataset_size)` at most; `1e-5` is a safe default |
| `dataset_size` | training rows on this client | **must be per-client**; drives the sampling rate q = batch/n |
| `batch_size` | local mini-batch size | must match what your learner actually uses |
| `local_epochs` | local epochs per round | must match your learner's setting |
| `budget_dir` | where to persist per-client JSON files | any writable local path on the client machine |

---

## Troubleshooting

**`ImportError: No module named 'budget_tracker'`**
The three files are not on the Python path. Make sure they are in the same
`custom/` folder as your other components, and that `custom/` is in
`PYTHONPATH` (NVFlare does this automatically for the standard layout).

**`DXO data_kind is not WEIGHT_DIFF`**
Your learner may be sending `WEIGHTS` (full model) instead of `WEIGHT_DIFF`
(delta). Change `DataKind.WEIGHT_DIFF` to `DataKind.WEIGHTS` in the
`process` method of `dp_gaussian_filter.py` and in the `_empty_shareable`
method. The clipping and noise logic is unchanged.

**Model accuracy collapses after adding DP**
Almost always a `clip_norm` problem. The norm of your weight updates is
probably much larger than `1.0`. Log `original_norm` from the debug output
and raise `clip_norm` to match the real distribution before re-running.

**`budget_exhausted` triggers unexpectedly early**
The accountant is conservative by design. If it fires too early, either
raise `target_epsilon` or increase `noise_multiplier`. Re-run
`dp_noise_calibrator.py` with the revised target to confirm the budget
still holds for the full 100 rounds.

**Resuming after a crash gives wrong ε**
Delete the stale JSON files in `budget_dir` and restart from round 0.
The accountant cannot safely resume a partial round — only complete rounds
are safe checkpoints.
