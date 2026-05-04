"""
Generate per-client diagnostics: anomaly score histograms and PR curves.

Usage (from the federated/ directory):
    uv run python gen_diagnostics_plots.py
    uv run python gen_diagnostics_plots.py --clients client0,client2
    uv run python gen_diagnostics_plots.py --model jobs/<job_id>/workspace/app_server/FL_global_model.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch_geometric.loader import LinkNeighborLoader

import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

# Make centralized utilities importable from the federated/ directory
_CENTRALIZED = os.path.join(os.path.dirname(__file__), "..", "centralized")
sys.path.insert(0, os.path.abspath(_CENTRALIZED))

from models.graphids import GraphIDS  # noqa: E402
from utils.trainers import test  # noqa: E402


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str, model: GraphIDS) -> GraphIDS:
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
    print(f"Loaded model from: {model_path}")
    return model


def build_test_loader(args, client_name: str):
    fed_clients_root = os.path.abspath(os.path.join(args.data_dir, "fed_clients"))
    client_dir = os.path.join(
        fed_clients_root, client_name, "pyg_graph_data", f"client_{client_name}"
    )
    test_graph = torch.load(os.path.join(client_dir, "test.pt"))[0]

    fanout_list = [args.fanout] if args.fanout != -1 else [-1]
    num_workers = min(os.cpu_count() or 0, 8)
    shuffle = args.positional_encoding == "None"

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
    return test_loader, test_graph


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


def plot_hist_and_pr(out_path: str, errors: np.ndarray, labels: np.ndarray) -> None:
    pos = errors[labels == 1]
    neg = errors[labels == 0]

    precision, recall, _ = precision_recall_curve(labels, errors)
    ap = average_precision_score(labels, errors)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(neg, bins=50, alpha=0.7, label="label 0", color="#4E79A7")
    axes[0].hist(pos, bins=50, alpha=0.7, label="label 1", color="#E15759")
    axes[0].set_title("Test score histogram")
    axes[0].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].plot(recall, precision, color="#59A14F")
    axes[1].set_title(f"PR curve (AP={ap:.4f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostics plots")
    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--ae_batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--fanout", type=int, default=-1)
    parser.add_argument("--clients", type=str, default="client0,client1,client2")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional global model checkpoint to evaluate for all clients.",
    )
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
    args = parser.parse_args()
    args.data_dir = DATASETS_DIR

    prod_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "poc_workspace", "fl_nids", "prod_00")
    )

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "diagnostics_plots"))
    os.makedirs(out_dir, exist_ok=True)

    clients = [c.strip() for c in args.clients.split(",") if c.strip()]

    for client_name in clients:
        print(f"\n=== {client_name} ===")
        test_loader, test_graph = build_test_loader(args, client_name)

        pe = None if args.positional_encoding == "None" else args.positional_encoding
        model = GraphIDS(
            ndim_in=test_graph.x.shape[1],
            edim_in=test_graph.edge_attr.shape[1],
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

        if args.model:
            model_path = os.path.abspath(args.model)
        else:
            model_path = find_client_checkpoint(prod_root, client_name)

        model = load_model(model_path, model)

        _, _, errors, labels, _ = test(
            model,
            test_loader,
            args.ae_batch_size,
            args.window_size,
            device,
            threshold=None,
        )

        errors_np = errors.cpu().numpy().astype(np.float64)
        labels_np = labels.cpu().numpy().astype(np.int64)

        pos_rate = float(np.mean(labels_np)) if labels_np.size else 0.0
        ap_pos = average_precision_score(labels_np, errors_np)
        ap_neg = average_precision_score(labels_np, -errors_np)
        print(f"Positive rate: {pos_rate:.6f}")
        print(f"AP (errors)  : {ap_pos:.6f}")
        print(f"AP (-errors) : {ap_neg:.6f}")

        out_path = os.path.join(out_dir, f"diagnostics_{client_name}.png")
        plot_hist_and_pr(out_path, errors_np, labels_np)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
