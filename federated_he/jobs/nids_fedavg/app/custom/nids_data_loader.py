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
):
    dataset = NetFlowDataset(
        name=dataset_name,
        data_dir=data_dir,
        force_reload=False,
        fraction=fraction,  # Only used if client_id is None
        data_type="benign",
        seed=seed,
        client_id=client_id,  # Use client_id from fed_clients
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