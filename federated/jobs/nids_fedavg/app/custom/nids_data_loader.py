import os
import torch
from torch_geometric.loader import LinkNeighborLoader
from utils.dataloaders import NetFlowDataset 


def get_loaders(
    data_dir: str,
    batch_size: int,
    fanout: int,
    dataset_name: str = "NF-CSE-CIC-IDS2018-v3",
    fraction: float = None,
    shuffle: bool = True,
    seed: int = 42,
    client_id: str = None,
    oversample_min_ratio: float = 0.05,
    oversample_target_ratio: float = 0.3,
    oversample_method: str = "borderline-1",
    oversample_random_state: int = 42,
):
    dataset = NetFlowDataset(
        name=dataset_name,
        data_dir=data_dir,
        force_reload=False,
        fraction=fraction,  # Only used if client_id is None
        data_type="benign",
        seed=seed,
        client_id=client_id,  # Use client_id from fed_clients
        oversample_min_ratio=oversample_min_ratio,
        oversample_target_ratio=oversample_target_ratio,
        oversample_method=oversample_method,
        oversample_random_state=oversample_random_state,
    )

    ndim_in = dataset.num_node_features
    edim_in = dataset.num_edge_features
    fanout_list = [fanout] if fanout != -1 else [-1]
    num_workers = min(os.cpu_count() or 0, 4)

    train_loader = LinkNeighborLoader(
        data=dataset.train_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.train_graph.edge_index,
        edge_label=dataset.train_graph.edge_labels,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    val_loader = LinkNeighborLoader(
        data=dataset.val_graph,
        num_neighbors=fanout_list,
        edge_label_index=dataset.val_graph.edge_index,
        edge_label=dataset.val_graph.edge_labels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader, ndim_in, edim_in