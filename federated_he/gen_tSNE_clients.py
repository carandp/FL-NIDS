import sys
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import torch
import tenseal as ts
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# Add custom federated app path for imports
FED_CUSTOM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "jobs/nids_fedavg/app/custom"))
if FED_CUSTOM_PATH not in sys.path:
    sys.path.insert(0, FED_CUSTOM_PATH)

from utils.dataloaders import NetFlowDataset
from graphids_model import GraphIDS


def find_he_context(model_path: str) -> str | None:
    """Find a TenSEAL context that includes the secret key for decryption."""
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "client0", "startup", "client_context.tenseal"),
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "client1", "startup", "client_context.tenseal"),
        os.path.join(script_dir, "poc_workspace", "fl_nids", "prod_00", "client2", "startup", "client_context.tenseal"),
        os.path.join(os.path.dirname(model_path), "client_context.tenseal"),
        os.path.join(os.path.dirname(model_path), "server_context.tenseal"),
        os.path.join(os.path.dirname(model_path), "..", "startup", "server_context.tenseal"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    return None


def decode_he_state_dict(state_dict: dict, model: GraphIDS, context_path: str) -> dict:
    """Decode HE-serialized bytes into tensors with expected parameter shapes."""
    with open(context_path, "rb") as f:
        ctx = ts.context_from(f.read())
    try:
        secret_key = ctx.secret_key()
    except Exception as e:
        raise RuntimeError(
            f"TenSEAL context at '{context_path}' does not include a secret key; use client_context.tenseal."
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


def run_tsne(Z, seed=42):
    Z = StandardScaler().fit_transform(Z)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=40,
        random_state=seed
    ).fit_transform(Z)


def plot_tsne_for_client(client_id):
    print(f"Processing t-SNE for {client_id}...")

    # 1) Load graph data for client (test split, federated)
    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../datasets/fed_clients/{client_id}"
    ))
    dataset = NetFlowDataset(
        name="NF-CSE-CIC-IDS2018-v3",
        data_dir=data_dir,
        force_reload=False,
        data_type="benign",
        seed=42,
        client_id=client_id,
    )
    graph = dataset.test_graph

    # 2) Build graph loader (single batch for t-SNE)
    graph_loader = [graph]

    # 3) Load model checkpoint and instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphIDS(
        ndim_in=dataset.num_node_features,
        edim_in=dataset.num_edge_features,
        edim_out=64,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        window_size=512,
        dropout=0.5,
        ae_dropout=0.2,
        positional_encoding=None,
        agg_type="mean",
        mask_ratio=0.15,
    )
    ckpt_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        f"../federated/poc_workspace/fl_nids/prod_00/{client_id}/checkpoints/best_global_model_{client_id}.pt"
    ))
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if any(isinstance(v, bytes) for v in state_dict.values()):
        context_path = find_he_context(ckpt_path)
        if not context_path:
            raise RuntimeError(
                "Checkpoint appears HE-serialized (bytes), but no client_context.tenseal file was found."
            )
        print(f"Detected HE checkpoint bytes. Decoding with context: {context_path}")
        state_dict = decode_he_state_dict(state_dict, model, context_path)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 4) Extract embeddings
    edge_embs = []
    recon_embs = []
    attacks = []
    with torch.no_grad():
        for batch in graph_loader:
            batch = batch.to(device)
            edge_embedding = model.encoder(
                batch.edge_index, batch.edge_attr, batch.edge_index.T, batch.num_nodes
            )
            edge_embs.append(edge_embedding.detach().cpu())
            recon = model.transformer(edge_embedding.unsqueeze(0)).squeeze(0)
            recon_embs.append(recon.detach().cpu())
            attacks.append(batch.edge_labels.detach().cpu())

    y_attack = np.concatenate(attacks) if len(attacks) > 0 else np.array([])
    H_edge = torch.cat(edge_embs, dim=0).numpy()
    H_rec = torch.cat(recon_embs, dim=0).numpy()

    # 5) Separate classes — subsample ONLY benign, keep ALL attacks
    label_map = {0: "Benign", 1: "Attack"}
    y_names = np.array([label_map.get(int(v), str(v)) for v in y_attack])

    benign_idx_all = np.where(y_names == "Benign")[0]
    attack_idx_all = np.where(y_names == "Attack")[0]

    max_benign = 10000  # tune this: lower = faster, higher = more faithful benign geometry
    rng = np.random.default_rng(42)

    if len(benign_idx_all) > max_benign:
        benign_idx_sampled = rng.choice(benign_idx_all, max_benign, replace=False)
    else:
        benign_idx_sampled = benign_idx_all

    print(f"  Benign sampled: {len(benign_idx_sampled):,} / {len(benign_idx_all):,} | Attacks (all): {len(attack_idx_all):,}")

    # Combine: subsampled benign + ALL attacks
    combined_idx = np.concatenate([benign_idx_sampled, attack_idx_all])
    combined_labels = y_names[combined_idx]

    # 6) t-SNE on combined set
    print("  Running t-SNE on edge embeddings...")
    Z_edge = run_tsne(H_edge[combined_idx])
    print("  Running t-SNE on reconstructed embeddings...")
    Z_rec = run_tsne(H_rec[combined_idx])

    # 7) Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    plots = [
        (axes[0], Z_edge, "Edge Embeddings (t-SNE)"),
        (axes[1], Z_rec,  "Reconstructed Edge Embeddings (t-SNE)"),
    ]

    # Use robust axis limits so a few extreme points do not squash the main cluster.
    def robust_limits(Z, low=0.05, high=99.95, pad=0.2):
        x_low, x_high = np.percentile(Z[:, 0], [low, high])
        y_low, y_high = np.percentile(Z[:, 1], [low, high])
        x_pad = (x_high - x_low) * pad
        y_pad = (y_high - y_low) * pad
        return (x_low - x_pad, x_high + x_pad), (y_low - y_pad, y_high + y_pad)

    for ax, Z, title in plots:
        benign_mask = combined_labels == "Benign"
        attack_mask = combined_labels == "Attack"
        attack_count = int(np.sum(attack_mask))
        benign_count = int(np.sum(benign_mask))

        # Hexbin density for benign
        hb = ax.hexbin(
            Z[benign_mask, 0],
            Z[benign_mask, 1],
            gridsize=55,
            cmap="Blues",
            mincnt=1,
            alpha=0.65
        )
        fig.colorbar(hb, ax=ax, label=f"Benign count (n={benign_count:,})")

        # Scatter all attack points
        ax.scatter(
            Z[attack_mask, 0],
            Z[attack_mask, 1],
            s=14,
            color="#ed8936",
            alpha=0.9,
            label=f"Attack (n={attack_count:,})"
        )

        # KDE contours over benign
        try:
            if np.sum(benign_mask) > 50:
                xy = np.vstack([Z[benign_mask, 0], Z[benign_mask, 1]])
                kde = gaussian_kde(xy)
                xx, yy = np.meshgrid(
                    np.linspace(Z[:, 0].min(), Z[:, 0].max(), 150),
                    np.linspace(Z[:, 1].min(), Z[:, 1].max(), 150)
                )
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, levels=6, colors="navy", alpha=0.4, linewidths=0.8)
        except Exception:
            pass

        ax.set_title(title)
        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.grid(True, linestyle="--", alpha=0.25)

        (xlim_low, xlim_high), (ylim_low, ylim_high) = robust_limits(Z)
        ax.set_xlim(xlim_low, xlim_high)
        ax.set_ylim(ylim_low, ylim_high)

        legend_elements = [
            Patch(facecolor="#2b6cb0", edgecolor="none", alpha=0.65, label=f"Benign (n={benign_count:,}, subsampled)"),
            Line2D([0], [0], marker='o', color='w', label=f"Attack (n={attack_count:,}, all)",
                   markerfacecolor="#ed8936", markersize=8)
        ]
        ax.legend(handles=legend_elements, title="Class", loc="upper center")

    plt.suptitle(f"t-SNE — {client_id}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join("tsne_plots", f"tSNE_{client_id}.png")
    os.makedirs("tsne_plots", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}\n")


if __name__ == "__main__":
    for client in ["client0", "client1", "client2"]:
        try:
            plot_tsne_for_client(client)
        except Exception as e:
            print(f"Failed for {client}: {e}")