
import sys
import os
# Add custom federated app path for imports
FED_CUSTOM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "jobs/nids_fedavg/app/custom"))
if FED_CUSTOM_PATH not in sys.path:
    sys.path.insert(0, FED_CUSTOM_PATH)

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Import graph loader and model

# Import federated custom data/model
from utils.dataloaders import NetFlowDataset
from graphids_model import GraphIDS



# -----------------------------
# 1) Load graph data for client0 (test split, federated)
# -----------------------------
dataset = NetFlowDataset(
    name="NF-CSE-CIC-IDS2018-v3",
    data_dir="/home/carandp/FL-NIDS/datasets/fed_clients/client0",
    force_reload=False,
    data_type="benign",
    seed=42,
    client_id="client0",
)
graph = dataset.test_graph


# -----------------------------
# 2) Build graph loader (single batch for t-SNE)
# -----------------------------
graph_loader = [graph]


# -----------------------------
# 3) Load model checkpoint and instantiate model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# These values should match your federated config
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


ckpt = torch.load("poc_workspace/fl_nids/prod_00/client0/checkpoints/best_global_model_client0.pt", map_location=device)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
elif "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"], strict=True)
else:
    model.load_state_dict(ckpt, strict=True)

model.to(device)
model.eval()




# -----------------------------
# 4) Extract embeddings
#    Adapt the forward call to your model
# -----------------------------

# --- Extract edge embeddings and their reconstructions ---
edge_embs = []
recon_embs = []
attacks = []

with torch.no_grad():
    for batch in graph_loader:
        batch = batch.to(device)

        # Edge embeddings: pass edge_couples as all edges
        edge_embedding = model.encoder(
            batch.edge_index, batch.edge_attr, batch.edge_index.T, batch.num_nodes
        )
        edge_embs.append(edge_embedding.detach().cpu())

        # Reconstructed (autoencoder) embeddings for edges
        # Pass edge embeddings through transformer (unsqueeze for batch dim)
        recon = model.transformer(edge_embedding.unsqueeze(0)).squeeze(0)
        recon_embs.append(recon.detach().cpu())

        # Use edge_labels as attack labels for plotting (for edge embeddings)
        attacks.append(batch.edge_labels.detach().cpu())



if len(attacks) > 0:
    y_attack = np.concatenate(attacks)
else:
    y_attack = np.array([])

# Prepare embedding matrices for t-SNE
H_edge = torch.cat(edge_embs, dim=0).numpy()
H_rec = torch.cat(recon_embs, dim=0).numpy()

# -----------------------------
# 5) t-SNE
# -----------------------------

def run_tsne(Z, seed=42):
    Z = StandardScaler().fit_transform(Z)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=seed
    ).fit_transform(Z)

Z_edge = run_tsne(H_edge)
Z_rec = run_tsne(H_rec)

# -----------------------------
# 6) Plot
# -----------------------------
classes = np.unique(y_attack)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]



# Plot edge embeddings and their reconstructions by attack type
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

for ax, Z, title in [
    (axes[0], Z_edge, "Edge Embeddings (t-SNE)"),
    (axes[1], Z_rec, "Reconstructed Edge Embeddings (t-SNE)")
]:
    for i, cls in enumerate(classes):
        idx = y_attack == cls
        ax.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.8, label=cls, color=colors[i % len(colors)])
    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(title="Attack Types", fontsize=8, title_fontsize=9, loc="upper center", ncol=4)

plt.tight_layout()
plt.show()