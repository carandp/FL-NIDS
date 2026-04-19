"""
Evaluate a federated FL_global_model.pt checkpoint produced by NVFlare's
PTFileModelPersistor.  Reuses the centralized test/validate utilities.

Usage (from the federated/ directory):
    uv run python gen_eval_federated.py \
        --model jobs/<job_id>/workspace/app_server/FL_global_model.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import tenseal as ts
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
from utils.trainers import find_threshold, test, validate  # noqa: E402
from torch_geometric.data import Data

device = "cuda" if torch.cuda.is_available() else "cpu"


def _find_he_context(model_path: str) -> str | None:
    """Try common locations for TenSEAL context that includes secret key."""
    script_dir = os.path.dirname(__file__)
    candidates = [
        # Client contexts include the secret key and can decrypt HE vectors.
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "client0", "startup", "client_context.tenseal"),
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "client1", "startup", "client_context.tenseal"),
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "client2", "startup", "client_context.tenseal"),
        # Active local POC workspace
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "server", "startup", "server_context.tenseal"),
        # Same tree as downloaded model, if copied there manually
        os.path.join(os.path.dirname(model_path), "client_context.tenseal"),
        os.path.join(os.path.dirname(model_path), "server_context.tenseal"),
        os.path.join(os.path.dirname(model_path), "..", "startup", "server_context.tenseal"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    return None


def _decode_he_state_dict(state_dict: dict, model: GraphIDS, context_path: str) -> dict:
    """Decode HE-serialized bytes into tensors matching model parameter shapes."""
    with open(context_path, "rb") as f:
        ctx = ts.context_from(f.read())
    try:
        secret_key = ctx.secret_key()
    except Exception as e:
        raise RuntimeError(
            f"TenSEAL context at '{context_path}' does not include a secret key; "
            "use a client_context.tenseal file for evaluation."
        ) from e

    expected_state = model.state_dict()
    decoded = {}
    for name, value in state_dict.items():
        if name not in expected_state:
            continue

        expected = expected_state[name]
        if isinstance(value, bytes):
            vec = ts.ckks_vector_from(ctx, value)
            plain = vec.decrypt(secret_key=secret_key)
            t = torch.tensor(plain, dtype=expected.dtype)
        else:
            t = torch.as_tensor(value, dtype=expected.dtype)

        if t.numel() != expected.numel():
            raise RuntimeError(
                f"Decoded param '{name}' has {t.numel()} values, expected {expected.numel()}"
            )
        decoded[name] = t.reshape(expected.shape)

    return decoded


def load_fl_model(model_path: str, model: GraphIDS) -> GraphIDS:
    """Load weights from an NVFlare PTFileModelPersistor checkpoint.

    The file is saved as:
        {"model": state_dict, "train_conf": ..., "meta_props": ...}
    """
    data = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = data["model"] if "model" in data else data
    # HE persistence stores tensors as serialized bytes. Detect and decode when needed.
    has_he_bytes = any(isinstance(v, bytes) for v in state_dict.values())
    if has_he_bytes:
        context_path = _find_he_context(model_path)
        if not context_path:
            raise RuntimeError(
                "Checkpoint appears HE-serialized (bytes), but no TenSEAL context file was found. "
                "Expected client_context.tenseal in federated_he/poc_workspace/fl_nids/prod_00/client*/startup "
                "or near the model file."
            )
        print(f"Detected HE checkpoint bytes. Decoding with context: {context_path}")
        state_dict = _decode_he_state_dict(state_dict, model, context_path)

    model.load_state_dict(state_dict)
    print(f"Loaded FL model from: {model_path}")
    return model


def load_and_merge_client_graphs(split, client_dirs):
    datas = []
    for client_dir in client_dirs:
        split_path = os.path.join(client_dir, f"{split}.pt")
        if os.path.exists(split_path):
            data = torch.load(split_path)[0]
            datas.append(data)
    # Merge edge_index, edge_attr, edge_labels, x, num_nodes
    # Assume all have same feature dimensions
    if not datas:
        raise RuntimeError(f"No data found for split {split} in clients: {client_dirs}")
    # Remap node indices to avoid collisions
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
    merged = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, num_nodes=x.shape[0])
    return merged

def build_loaders(args):
    fed_clients_root = os.path.abspath(os.path.join(args.data_dir, "fed_clients"))
    client_dirs = [
        os.path.join(fed_clients_root, f"client{i}", "pyg_graph_data", f"client_client{i}")
        for i in range(3)
    ]
    val_graph = load_and_merge_client_graphs("val", client_dirs)
    test_graph = load_and_merge_client_graphs("test", client_dirs)
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



def main():

    parser = argparse.ArgumentParser(description="Evaluate a federated FL_global_model.pt")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to FL_global_model.pt produced by NVFlare PTFileModelPersistor",
    )
    # Use datasets directory at the project root
    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
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
    args = parser.parse_args()
    args.data_dir = DATASETS_DIR
    PROD_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "poc_workspace", "fl_nids", "prod_00")
    )

    # Find next available report file name
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports"))
    os.makedirs(reports_dir, exist_ok=True)
    n = 0
    while True:
        report_path = os.path.join(reports_dir, f"report_{n}.txt")
        if not os.path.exists(report_path):
            break
        n += 1

    class FileOnlyWriter:
        def __init__(self, file):
            self.file = file
        def write(self, msg):
            self.file.write(msg)
        def flush(self):
            self.file.flush()

    with open(report_path, "w") as f:
        # Redirect stdout to file only
        old_stdout = sys.stdout
        sys.stdout = FileOnlyWriter(f)
        try:
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

            print_all_metrics(PROD_ROOT)
        finally:
            sys.stdout = old_stdout

    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    main()
