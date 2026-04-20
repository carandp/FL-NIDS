
import sys
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import torch
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

def run_tsne(Z, seed=42):
    Z = StandardScaler().fit_transform(Z)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=seed
    ).fit_transform(Z)

def plot_tsne_for_client(client_id):
    print(f"Processing t-SNE for {client_id}...")
    # 1) Load graph data for client (test split, federated)
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../datasets/fed_clients/{client_id}"))
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
        edim_out=64,  # from config
        embed_dim=32,  # ae_embedding_dim from config
        num_heads=1,  # adjust if needed
        num_layers=1,
        window_size=512,
        dropout=0.5,
        ae_dropout=0.2,
        positional_encoding=None,
        agg_type="mean",
        mask_ratio=0.15,
    )
    ckpt_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), f"../federated_dp/poc_workspace/fl_nids/prod_00/{client_id}/checkpoints/best_global_model_{client_id}.pt"
    ))
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
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

    if len(attacks) > 0:
        y_attack = np.concatenate(attacks)
    else:
        y_attack = np.array([])
    H_edge = torch.cat(edge_embs, dim=0).numpy()
    H_rec = torch.cat(recon_embs, dim=0).numpy()

    # 5) t-SNE
    Z_edge = run_tsne(H_edge)
    Z_rec = run_tsne(H_rec)

    # 6) Plot
    label_map = {0: "Benign", 1: "Attack"}
    y_names = np.array([label_map.get(int(v), str(v)) for v in y_attack])

    # Downsample + Balance
    max_per_class = 4000
    indices = []
    for cls in np.unique(y_names):
        cls_idx = np.where(y_names == cls)[0]
        if len(cls_idx) > max_per_class:
            cls_idx = np.random.choice(cls_idx, max_per_class, replace=False)
        indices.extend(cls_idx)
    indices = np.array(indices)
    Z_edge = Z_edge[indices]
    Z_rec = Z_rec[indices]
    y_names = y_names[indices]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    plots = [
        (axes[0], Z_edge, "Edge Embeddings (t-SNE)"),
        (axes[1], Z_rec, "Reconstructed Edge Embeddings (t-SNE)")
    ]
    for ax, Z, title in plots:
        benign_idx = y_names == "Benign"
        attack_idx = y_names == "Attack"
        hb = ax.hexbin(
            Z[benign_idx, 0],
            Z[benign_idx, 1],
            gridsize=55,
            cmap="Blues",
            mincnt=1,
            alpha=0.65
        )
        ax.scatter(
            Z[attack_idx, 0],
            Z[attack_idx, 1],
            s=14,
            color="#ed8936",
            alpha=0.9,
            label="Attack"
        )
        try:
            if np.sum(benign_idx) > 50:
                xy = np.vstack([Z[benign_idx, 0], Z[benign_idx, 1]])
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
        legend_elements = [
            Patch(facecolor="#2b6cb0", edgecolor="none", alpha=0.65, label="Benign"),
            Line2D([0], [0], marker='o', color='w', label='Attack',
                markerfacecolor="#ed8936", markersize=8)
        ]
        ax.legend(handles=legend_elements, title="Class", loc="upper center")
    plt.tight_layout()
    out_path = os.path.join("tsne_plots", f"tSNE_{client_id}.png")
    os.makedirs("tsne_plots", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot for {client_id} to {out_path}")

if __name__ == "__main__":
    for client in ["client0", "client1", "client2"]:
        try:
            plot_tsne_for_client(client)
        except Exception as e:
            print(f"Failed for {client}: {e}")