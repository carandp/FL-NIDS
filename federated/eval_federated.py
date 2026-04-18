"""
Evaluate a federated FL_global_model.pt checkpoint produced by NVFlare's
PTFileModelPersistor.  Reuses the centralized test/validate utilities.

Usage (from the federated/ directory):
    uv run python eval_federated.py \
        --model path/to/FL_global_model.pt \
        --dataset NF-CSE-CIC-IDS2018-v3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader

import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

_BLOCKS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list) -> str:
    """Map a list of floats to an 8-level Unicode block sparkline."""
    if not values:
        return ""
    if len(values) == 1:
        return _BLOCKS[0]
    lo, hi = min(values), max(values)
    if hi == lo:
        return _BLOCKS[0] * len(values)
    return "".join(_BLOCKS[round((v - lo) / (hi - lo) * 7)] for v in values)


def print_run_history(metrics_path: str) -> None:
    """Print a wandb-style run history and summary from a metrics JSON file."""
    with open(metrics_path) as _f:
        history = json.load(_f)
    if not history:
        return

    keys = ["train_loss", "val_loss", "val_pr_auc", "val_f1"]
    col_w = max(len(k) for k in keys)
    site = os.path.basename(metrics_path).replace("metrics_history_", "").replace(".json", "")

    print(f"\nRun history ({site}):")
    for k in keys:
        vals = [r[k] for r in history if k in r]
        print(f"  {k:>{col_w}} {_sparkline(vals)}")

    last = history[-1]
    print("\nRun summary:")
    for k in keys:
        if k in last:
            print(f"  {k:>{col_w}} {last[k]:.6g}")


def print_all_metrics(prod_root: str) -> None:
    """Read per-site metrics from prod_00/site-*/checkpoints without glob searching."""
    if not os.path.isdir(prod_root):
        return

    # Print metrics for all subdirectories with a metrics_history_<site>.json file in their checkpoints folder
    for site_name in sorted(os.listdir(prod_root)):
        site_path = os.path.join(prod_root, site_name)
        if not os.path.isdir(site_path):
            continue
        metrics_path = os.path.join(site_path, "checkpoints", f"metrics_history_{site_name}.json")
        if os.path.isfile(metrics_path):
            print_run_history(metrics_path)


# Make centralized utilities importable from the federated/ directory
_CENTRALIZED = os.path.join(os.path.dirname(__file__), "..", "centralized")
sys.path.insert(0, os.path.abspath(_CENTRALIZED))

from models.graphids import GraphIDS  # noqa: E402
from utils.dataloaders import NetFlowDataset  # noqa: E402
from utils.trainers import find_threshold, test, validate  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_fl_model(model_path: str, model: GraphIDS) -> GraphIDS:
    """Load weights from an NVFlare PTFileModelPersistor checkpoint.

    The file is saved as:
        {"model": state_dict, "train_conf": ..., "meta_props": ...}
    """
    data = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = data["model"] if "model" in data else data
    model.load_state_dict(state_dict)
    print(f"Loaded FL model from: {model_path}")
    return model


def build_loaders(args):
    dataset = NetFlowDataset(
        name=args.dataset,
        data_dir=args.data_dir,
        force_reload=args.reload_dataset,
        fraction=args.fraction,
        data_type=args.data_type,
        seed=args.seed,
    )

    ndim_in = dataset.num_node_features
    edim_in = dataset.num_edge_features
    print(f"Node features: {ndim_in}  |  Edge features: {edim_in}")

    fanout_list = [args.fanout] if args.fanout != -1 else [-1]
    num_workers = min(os.cpu_count() or 0, 8)
    shuffle = args.positional_encoding == "None"

    val_loader = LinkNeighborLoader(
        data=dataset.val_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.val_graph.edge_index,
        edge_label=dataset.val_graph.edge_labels,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    test_loader = LinkNeighborLoader(
        data=dataset.test_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.test_graph.edge_index,
        edge_label=dataset.test_graph.edge_labels,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return val_loader, test_loader, ndim_in, edim_in


def main():
    parser = argparse.ArgumentParser(description="Evaluate a federated FL_global_model.pt")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to FL_global_model.pt produced by NVFlare PTFileModelPersistor",
    )
    # Use datasets directory at the project root
    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    parser.add_argument(
        "--dataset", type=str, default="NF-CSE-CIC-IDS2018-v3",
        choices=["NF-UNSW-NB15-v3", "NF-CSE-CIC-IDS2018-v3",
                 "NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2"],
    )
    parser.add_argument("--data_type", type=str, default="benign",
                        choices=["benign", "mixed"])
    parser.add_argument("--fraction", type=float, default=0.2,
                        help="Fraction of the dataset to load (default: 0.2 to reuse the FL training cache; "
                             "pass a different value + --reload_dataset to reprocess)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--ae_batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--fanout", type=int, default=-1)
    # Model architecture — must match the server config
    parser.add_argument("--edim_out", type=int, default=64)
    parser.add_argument("--ae_embedding_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--ae_dropout", type=float, default=0.2)
    parser.add_argument("--agg_type", type=str, default="mean")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--positional_encoding", type=str, default="None",
                        choices=["None", "learnable", "sinusoidal"])
    parser.add_argument("--threshold_method", type=str, default="supervised",
                        choices=["supervised", "unsupervised"],
                        help="How to derive the anomaly threshold from the validation set")
    parser.add_argument("--save_curve", action="store_true",
                        help="Save precision-recall curve as .npz")
    parser.add_argument("--reload_dataset", action="store_true")
    args = parser.parse_args()
    args.data_dir = DATASETS_DIR
    PROD_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "poc_workspace", "fl_nids", "prod_00")
    )

    # ------------------------------------------------------------------ #
    # 1. Build data loaders
    # ------------------------------------------------------------------ #
    val_loader, test_loader, ndim_in, edim_in = build_loaders(args)

    # ------------------------------------------------------------------ #
    # 2. Build model and load FL weights
    # ------------------------------------------------------------------ #
    pe = None if args.positional_encoding == "None" else args.positional_encoding
    model = GraphIDS(
        ndim_in=ndim_in,
        edim_in=edim_in,
        edim_out=args.edim_out,
        embed_dim=args.ae_embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        window_size=args.window_size,
        dropout=args.dropout,
        ae_dropout=args.ae_dropout,
        positional_encoding=pe,
        agg_type=args.agg_type,
        mask_ratio=args.mask_ratio,
    ).to(device)

    model = load_fl_model(args.model, model)

    # ------------------------------------------------------------------ #
    # 3. Derive threshold from validation set
    # ------------------------------------------------------------------ #
    print("Computing validation errors to derive threshold...")
    _, val_errors, val_labels = validate(
        model, val_loader, args.ae_batch_size, args.window_size, device
    )
    threshold = find_threshold(val_errors, val_labels, method=args.threshold_method)
    print(f"Threshold ({args.threshold_method}): {threshold:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate on test set
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Evaluating on test set...")
    f1, pr_auc, errors, test_labels, pred_time = test(
        model, test_loader, args.ae_batch_size, args.window_size, device,
        threshold=threshold,
    )

    print(f"\nTest macro F1-score : {f1:.4f}")
    print(f"Test PR-AUC         : {pr_auc:.4f}")
    print(f"Prediction time     : {pred_time:.4f} s")

    if args.save_curve:
        precision, recall, _ = precision_recall_curve(test_labels.cpu(), errors.cpu())
        out_dir = os.path.join(os.path.dirname(__file__), "curves")
        os.makedirs(out_dir, exist_ok=True)
        curve_path = os.path.join(out_dir, f"precision_recall_{args.dataset}_federated.npz")
        np.savez(curve_path, precision=precision, recall=recall)
        print(f"PR curve saved to   : {curve_path}")

    print_all_metrics(PROD_ROOT)

if __name__ == "__main__":
    main()
