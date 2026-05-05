"""
Generate global diagnostics: joint anomaly score histogram and PR curve.

Usage (from the federated/ directory):
    uv run python gen_diagnostic_plot_global.py --job <job_id>
    uv run python gen_diagnostic_plot_global.py --job <job_id> --clients client0,client2
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
from torch_geometric.data import Data
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


def load_and_merge_client_graphs(split: str, client_dirs: list[str]) -> Data:
    datas = []
    for client_dir in client_dirs:
        split_path = os.path.join(client_dir, f"{split}.pt")
        if os.path.exists(split_path):
            data = torch.load(split_path)[0]
            datas.append(data)

    if not datas:
        raise RuntimeError(f"No data found for split {split} in clients: {client_dirs}")

    node_offset = 0
    edge_index_list, edge_attr_list, edge_labels_list = [], [], []
    x_list = []
    for data in datas:
        num_nodes = data.num_nodes
        edge_index_list.append(data.edge_index + node_offset)
        edge_attr_list.append(data.edge_attr)
        edge_labels_list.append(data.edge_labels)
        x_list.append(data.x)
        node_offset += num_nodes

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    edge_labels = torch.cat(edge_labels_list, dim=0)
    x = torch.cat(x_list, dim=0)
    merged = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_labels=edge_labels,
        num_nodes=x.shape[0],
    )
    return merged


def build_test_loader(args, clients: list[str]):
    fed_clients_root = os.path.abspath(os.path.join(args.data_dir, "fed_clients"))
    client_dirs = [
        os.path.join(fed_clients_root, client, "pyg_graph_data", f"client_{client}")
        for client in clients
    ]
    test_graph = load_and_merge_client_graphs("test", client_dirs)

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


def plot_hist_and_pr(
    out_path: str,
    errors: np.ndarray,
    labels: np.ndarray,
    hist_min: float,
    hist_max: float,
) -> None:
    pos = errors[labels == 1]
    neg = errors[labels == 0]

    precision, recall, _ = precision_recall_curve(labels, errors)
    ap = average_precision_score(labels, errors)
    ap_inverted = average_precision_score(labels, -errors)
    pos_rate = float(np.mean(labels)) if labels.size else 0.0

    print(f"Positive rate: {pos_rate:.6f}")
    print(f"AP(errors): {ap:.6f}")
    print(f"AP(-errors): {ap_inverted:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(neg, bins=1000, alpha=0.7, label="label 0", color="#4E79A7")
    axes[0].hist(pos, bins=1000, alpha=0.7, label="label 1", color="#E15759")
    axes[0].set_title("Test score histogram")
    axes[0].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(hist_min, hist_max)
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
    parser = argparse.ArgumentParser(description="Generate global diagnostics plot")
    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    parser.add_argument("--job", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--ae_batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--fanout", type=int, default=-1)
    parser.add_argument("--clients", type=str, default="client0,client1,client2")
    parser.add_argument("--hist_min", type=float, default=0.0)
    parser.add_argument("--hist_max", type=float, default=0.02)
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

    job_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "jobs", args.job))
    model_path = os.path.join(job_dir, "workspace", "app_server", "FL_global_model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing global model at: {model_path}")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "diagnostics_plots"))
    os.makedirs(out_dir, exist_ok=True)

    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    print(f"Clients: {', '.join(clients)}")

    test_loader, test_graph = build_test_loader(args, clients)

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

    out_path = os.path.join(out_dir, f"diagnostics_global_{args.job}.png")
    plot_hist_and_pr(out_path, errors_np, labels_np, args.hist_min, args.hist_max)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
