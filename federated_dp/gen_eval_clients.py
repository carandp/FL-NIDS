"""
Evaluate per-client checkpoints and report metrics for each client.

Usage (from the federated/ directory):
    uv run python gen_eval_clients.py
"""

import argparse
import json
import os
import sys

import torch
from torch_geometric.loader import LinkNeighborLoader

import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

_SPARK = " .:-=+*#%@"


def _sparkline(values: list[float]) -> str:
    """Map a list of floats to a simple ASCII sparkline."""
    if not values:
        return ""
    if len(values) == 1:
        return _SPARK[1]
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK[1] * len(values)
    out = []
    for v in values:
        idx = round((v - lo) / (hi - lo) * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


def print_run_history(metrics_path: str) -> None:
    """Print a wandb-style run history and summary from a metrics JSON file."""
    if not os.path.isfile(metrics_path):
        return
    with open(metrics_path) as _f:
        history = json.load(_f)
    if not history:
        return

    keys = ["train_loss", "val_loss", "val_pr_auc", "val_macro_f1"]
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


# Make centralized utilities importable from the federated/ directory
_CENTRALIZED = os.path.join(os.path.dirname(__file__), "..", "centralized")
sys.path.insert(0, os.path.abspath(_CENTRALIZED))

from models.graphids import GraphIDS  # noqa: E402
from utils.trainers import find_threshold, test, validate  # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_local_model(model_path: str, model: GraphIDS) -> GraphIDS:
    """Load weights from a per-client checkpoint."""
    data = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(data, dict):
        if "model_state_dict" in data:
            state_dict = data["model_state_dict"]
        elif "model" in data:
            state_dict = data["model"]
        else:
            state_dict = data
    else:
        state_dict = data
    model.load_state_dict(state_dict)
    print(f"Loaded client model from: {model_path}")
    return model


def build_loaders_for_client(args, client_name: str):
    fed_clients_root = os.path.abspath(os.path.join(args.data_dir, "fed_clients"))
    client_dir = os.path.join(
        fed_clients_root, client_name, "pyg_graph_data", f"client_{client_name}"
    )
    val_graph = torch.load(os.path.join(client_dir, "val.pt"))[0]
    test_graph = torch.load(os.path.join(client_dir, "test.pt"))[0]
    ndim_in = val_graph.x.shape[1]
    edim_in = val_graph.edge_attr.shape[1]
    print(f"Node features: {ndim_in}  |  Edge features: {edim_in}")

    fanout_list = [args.fanout] if args.fanout != -1 else [-1]
    num_workers = min(os.cpu_count() or 0, 8)
    shuffle = args.positional_encoding == "None"

    val_loader = LinkNeighborLoader(
        data=val_graph,
        num_neighbors=fanout_list,
        edge_label_index=val_graph.edge_index,
        edge_label=val_graph.edge_labels,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    test_loader = LinkNeighborLoader(
        data=test_graph,
        num_neighbors=fanout_list,
        edge_label_index=test_graph.edge_index,
        edge_label=test_graph.edge_labels,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return val_loader, test_loader, ndim_in, edim_in


def find_client_checkpoint(prod_root: str, client_name: str) -> str:
    ckpt = os.path.join(
        prod_root,
        client_name,
        "checkpoints",
        f"best_global_model_{client_name}.pt",
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-client checkpoints")
    # Use datasets directory at the project root
    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--ae_batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--fanout", type=int, default=-1)
    parser.add_argument("--clients", type=str, default="client0,client1,client2")
    # Model architecture — must match the server config
    parser.add_argument("--edim_out", type=int, default=64)
    parser.add_argument("--ae_embedding_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--ae_dropout", type=float, default=0.2)
    parser.add_argument("--agg_type", type=str, default="mean")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default="None",
        choices=["None", "learnable", "sinusoidal"],
    )
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="supervised",
        choices=["supervised", "unsupervised"],
        help="How to derive the anomaly threshold from the validation set",
    )
    args = parser.parse_args()
    args.data_dir = DATASETS_DIR

    prod_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "poc_workspace", "fl_nids", "prod_00")
    )

    # Find next available report file name
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    n = 0
    while True:
        report_path = os.path.join(reports_dir, f"report_clients_{n}.txt")
        if not os.path.exists(report_path):
            break
        n += 1

    clients = [c.strip() for c in args.clients.split(",") if c.strip()]

    class FileOnlyWriter:
        def __init__(self, file):
            self.file = file
        def write(self, msg):
            self.file.write(msg)
        def flush(self):
            self.file.flush()

    with open(report_path, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = FileOnlyWriter(f)
        try:
            for client_name in clients:
                print(f"\n=== {client_name} ===")

                val_loader, test_loader, ndim_in, edim_in = build_loaders_for_client(
                    args, client_name
                )

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

                model_path = find_client_checkpoint(prod_root, client_name)
                model = load_local_model(model_path, model)

                print("Computing validation errors to derive threshold...")
                _, val_errors, val_labels = validate(
                    model, val_loader, args.ae_batch_size, args.window_size, device
                )
                threshold = find_threshold(
                    val_errors, val_labels, method=args.threshold_method
                )
                print(f"Threshold ({args.threshold_method}): {threshold:.6f}")

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                print("Evaluating on test set...")
                f1, pr_auc, _, _, pred_time = test(
                    model,
                    test_loader,
                    args.ae_batch_size,
                    args.window_size,
                    device,
                    threshold=threshold,
                )

                print(f"\nTest macro F1-score : {f1:.4f}")
                print(f"Test PR-AUC         : {pr_auc:.4f}")
                print(f"Prediction time     : {pred_time:.4f} s")

                metrics_path = os.path.join(
                    prod_root,
                    client_name,
                    "checkpoints",
                    f"metrics_history_{client_name}.json",
                )
                print_run_history(metrics_path)
        finally:
            sys.stdout = old_stdout

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
