import inspect
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,  # type: ignore[attr-defined]
        torch_geometric.data.data.DataTensorAttr,  # type: ignore[attr-defined]
        torch_geometric.data.storage.GlobalStorage,  # type: ignore[attr-defined]
    ]
)


def collate_fn(batch):
    sequences, masks = zip(*batch, strict=False)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
    return sequences_padded, masks_padded


class SequentialDataset(TorchDataset):
    def __init__(self, data, window, device, step=None):
        self.data = data
        self.window = window
        self.device = device
        if step is None:
            self.step = window
        else:
            self.step = step

    def __getitem__(self, index):
        start_idx = index * self.step
        end_idx = min(start_idx + self.window, len(self.data))
        x = self.data[start_idx:end_idx].to(self.device)
        mask = torch.ones_like(x, dtype=torch.bool).to(self.device)
        return x, mask

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.step + 1)


class NetFlowDataset:
    def __init__(
        self,
        name,
        data_dir,
        force_reload=False,
        fraction=None,
        data_type="benign",
        seed=42,
        client_id=None,
        oversample_min_ratio=0.05,
        oversample_target_ratio=0.3,
        oversample_method="borderline-1",
        oversample_random_state=42,
    ):
        self.name = name
        self.data_dir = data_dir
        self.fraction = fraction
        self.data_type = data_type
        self.seed = seed
        self.client_id = client_id
        self.oversample_min_ratio = oversample_min_ratio
        self.oversample_target_ratio = oversample_target_ratio
        self.oversample_method = oversample_method
        self.oversample_random_state = oversample_random_state

        # Setup directories
        graph_dir = os.path.join(data_dir, "pyg_graph_data")
        
        oversample_tag = None
        if self.oversample_min_ratio is not None and self.oversample_target_ratio is not None:
            min_tag = int(self.oversample_min_ratio * 100)
            target_tag = int(self.oversample_target_ratio * 100)
            oversample_tag = f"os{min_tag}_{target_tag}_{self.oversample_method}"

        # When loading from fed_clients, use client_id in directory name
        if client_id is not None:
            base_dir = os.path.join(graph_dir, f"client_{client_id}")
        elif fraction is not None:
            assert 0 < fraction < 1
            fraction_str = str(fraction).replace(".", "_")
            base_dir = os.path.join(graph_dir, f"{name}_{fraction_str}")
        else:
            base_dir = os.path.join(graph_dir, name)

        if oversample_tag:
            self.processed_dir = f"{base_dir}_{oversample_tag}"
        else:
            self.processed_dir = base_dir

        self.raw_dir = os.path.join(data_dir, name)

        # Handle force reload
        if force_reload and os.path.exists(self.processed_dir):
            print(
                f"Force reload: Removing existing processed data at {self.processed_dir}"
            )
            shutil.rmtree(self.processed_dir)

            # Also remove the scaler for this dataset
            scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
            if os.path.exists(scaler_path):
                print(f"Removing old scaler: {scaler_path}")
                os.remove(scaler_path)

        # Check if we need to process
        if self._needs_processing():
            # Check for seed mismatch before processing
            seed_file = os.path.join(self.processed_dir, ".seed")
            if os.path.exists(seed_file):
                with open(seed_file) as f:
                    cached_seed = int(f.read().strip())
                if cached_seed != self.seed:
                    print(
                        f"Warning: Cached data was created with seed={cached_seed}, but current seed={self.seed}"
                    )
                    print(
                        "Run with --reload_dataset to recreate data with the new seed"
                    )

            self._process()
        else:
            # Check for seed mismatch when loading existing cache
            seed_file = os.path.join(self.processed_dir, ".seed")
            if os.path.exists(seed_file):
                with open(seed_file) as f:
                    cached_seed = int(f.read().strip())
                if cached_seed != self.seed:
                    print(
                        f"Warning: Cached data was created with seed={cached_seed}, but current seed={self.seed}"
                    )
                    print(
                        "Run with --reload_dataset to recreate data with the new seed"
                    )

        # Load the processed data
        self.train_graph = torch.load(os.path.join(self.processed_dir, "train.pt"))[0]
        self.val_graph = torch.load(os.path.join(self.processed_dir, "val.pt"))[0]
        self.test_graph = torch.load(os.path.join(self.processed_dir, "test.pt"))[0]

    def _needs_processing(self):
        """Check if processing is needed"""
        if not os.path.exists(self.processed_dir):
            return True

        required_files = ["train.pt", "val.pt", "test.pt"]
        for filename in required_files:
            if not os.path.exists(os.path.join(self.processed_dir, filename)):
                return True

        return False

    def _process(self):
        """Process the raw data (CSV or Parquet) and create train/val/test splits"""
        print(f"Processing dataset {self.name}...")

        os.makedirs(self.processed_dir, exist_ok=True)

        # Load data from parquet (fed_clients) or CSV (raw directory)
        if self.client_id is not None:
            # Load from fed_clients parquet
            parquet_path = os.path.join(self.data_dir, f"{self.client_id}.parquet")
            print(f"Loading parquet from: {parquet_path}")
            df = pd.read_parquet(parquet_path, engine="fastparquet")
        else:
            # Load from raw CSV with chunking
            csv_path = os.path.join(self.raw_dir, f"{self.name}.csv")
            timestamp_cols = {"FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"}
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=500_000):
                for col in chunk.select_dtypes(include=["float64"]).columns:
                    chunk[col] = chunk[col].astype(np.float32)
                for col in chunk.select_dtypes(include=["int64"]).columns:
                    if col in timestamp_cols:
                        continue
                    elif col == "Label":
                        chunk[col] = chunk[col].astype(np.int8)
                    else:
                        chunk[col] = chunk[col].astype(np.int32)
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            del chunks

            if self.fraction is not None:
                df = df.groupby(by="Attack").sample(
                    frac=self.fraction, random_state=self.seed
                )

        x = df.drop(columns=["Attack", "Label"], errors="ignore")
        y = df[["Attack", "Label"]] if all(col in df.columns for col in ["Attack", "Label"]) else None

        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        if "v3" in self.name:
            edge_features = [
                col
                for col in x.columns
                if col
                not in [
                    "IPV4_SRC_ADDR",
                    "IPV4_DST_ADDR",
                    "FLOW_END_MILLISECONDS",
                    "FLOW_START_MILLISECONDS",
                ]
            ]
        else:
            edge_features = [
                col
                for col in x.columns
                if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
            ]

        # Reconstruct df with features and labels
        if y is not None:
            df = pd.concat([x, y], axis=1)
            stratify_col = y["Attack"]
        else:
            # For parquet from fed_clients, create Attack and Label columns if missing
            df = x.copy()
            if "Label" not in df.columns:
                df["Label"] = 0
            if "Attack" not in df.columns:
                df["Attack"] = "Benign"
            stratify_col = df["Attack"]

        df_train, df_val_test = train_test_split(
            df, test_size=0.2, random_state=self.seed, stratify=stratify_col
        )

        if self.data_type == "benign":
            df_train = df_train[df_train["Label"] == 0]

        scaler_path = os.path.join("scalers", f"scaler_{self.name}.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            except Exception as e:
                print(f"Failed to load scaler: {e}. Creating new one.")
                scaler = MinMaxScaler()
                scaler.fit(df_train[edge_features])
        else:
            scaler = MinMaxScaler()
            scaler.fit(df_train[edge_features])
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        df_train[edge_features] = scaler.transform(df_train[edge_features])
        df_val_test_scaled = scaler.transform(df_val_test[edge_features])
        df_val_test[edge_features] = np.clip(df_val_test_scaled, -10, 10)

        df_val, df_test = train_test_split(
            df_val_test,
            test_size=0.5,
            random_state=self.seed,
            stratify=df_val_test["Attack"],
        )

        df_train = self._maybe_oversample_train(df_train, edge_features)

        if "v3" in self.name:
            df_train = df_train.sort_values(by="FLOW_START_MILLISECONDS")
            df_val = df_val.sort_values(by="FLOW_START_MILLISECONDS")
            df_test = df_test.sort_values(by="FLOW_START_MILLISECONDS")

        unique_nodes = pd.concat(
            [
                df_train["IPV4_SRC_ADDR"],
                df_train["IPV4_DST_ADDR"],
                df_val["IPV4_SRC_ADDR"],
                df_val["IPV4_DST_ADDR"],
                df_test["IPV4_SRC_ADDR"],
                df_test["IPV4_DST_ADDR"],
            ]
        ).unique()
        node_map = {node: i for i, node in enumerate(unique_nodes)}
        num_nodes = len(node_map)

        datasets = {"train": df_train, "val": df_val, "test": df_test}

        for split_name, df_split in datasets.items():
            src_nodes = np.array([node_map[ip] for ip in df_split["IPV4_SRC_ADDR"]])
            dst_nodes = np.array([node_map[ip] for ip in df_split["IPV4_DST_ADDR"]])
            edge_index = torch.tensor(
                np.array([src_nodes, dst_nodes]), dtype=torch.long
            )
            edge_attr = torch.tensor(df_split[edge_features].values, dtype=torch.float)
            edge_labels = torch.tensor(df_split["Label"].values, dtype=torch.long)
            x = torch.ones(num_nodes, edge_attr.shape[1], dtype=torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_labels=edge_labels,
                num_nodes=num_nodes,
            )

            # Save as list for compatibility
            torch.save([data], os.path.join(self.processed_dir, f"{split_name}.pt"))

        # Save seed information for cache validation
        seed_file = os.path.join(self.processed_dir, ".seed")
        with open(seed_file, "w") as f:
            f.write(str(self.seed))

        print("Done!")

    def _maybe_oversample_train(self, df_train, edge_features):
        if self.oversample_min_ratio is None or self.oversample_target_ratio is None:
            return df_train

        if "Label" not in df_train.columns:
            return df_train

        label_counts = df_train["Label"].value_counts()
        if len(label_counts) < 2:
            return df_train

        minority_label = label_counts.idxmin()
        minority_ratio = label_counts.min() / label_counts.sum()
        if minority_ratio >= self.oversample_min_ratio:
            return df_train

        try:
            from imblearn.over_sampling import BorderlineSMOTE
        except ImportError:
            print(
                "Imbalanced-learn is not installed; skipping local oversampling. "
                "Install imbalanced-learn to enable SMOTE-based balancing."
            )
            return df_train

        smote_kwargs = {
            "sampling_strategy": self.oversample_target_ratio,
            "random_state": self.oversample_random_state,
        }
        if "kind" in inspect.signature(BorderlineSMOTE).parameters:
            smote_kwargs["kind"] = self.oversample_method

        smote = BorderlineSMOTE(**smote_kwargs)

        x = df_train[edge_features].to_numpy()
        y = df_train["Label"].to_numpy()
        x_resampled, y_resampled = smote.fit_resample(x, y)

        if len(x_resampled) == len(df_train):
            return df_train

        df_features = pd.DataFrame(x_resampled, columns=edge_features)
        labels_resampled = pd.Series(y_resampled, name="Label").astype(df_train["Label"].dtype)

        passthrough_cols = [
            col for col in df_train.columns if col not in edge_features + ["Label"]
        ]
        if passthrough_cols:
            num_new = len(df_features) - len(df_train)
            if num_new > 0:
                sample_rows = (
                    df_train[df_train["Label"] == minority_label][passthrough_cols]
                    .sample(
                        n=num_new,
                        replace=True,
                        random_state=self.oversample_random_state,
                    )
                    .reset_index(drop=True)
                )
                passthrough_resampled = pd.concat(
                    [df_train[passthrough_cols].reset_index(drop=True), sample_rows],
                    ignore_index=True,
                )
            else:
                passthrough_resampled = df_train[passthrough_cols].reset_index(drop=True)

            df_train = pd.concat(
                [df_features, labels_resampled, passthrough_resampled], axis=1
            )
        else:
            df_train = pd.concat([df_features, labels_resampled], axis=1)

        new_counts = df_train["Label"].value_counts()
        new_ratio = new_counts.min() / new_counts.sum()
        print(
            "Applied local oversampling: "
            f"minority_ratio {minority_ratio:.4f} -> {new_ratio:.4f} "
            f"(target={self.oversample_target_ratio})"
        )
        return df_train

    def __len__(self):
        # Return total number of graphs (for compatibility)
        return 3

    @property
    def num_node_features(self):
        return self.train_graph.x.shape[1]

    @property
    def num_edge_features(self):
        return self.train_graph.edge_attr.shape[1]

    @property
    def num_nodes(self):
        return self.train_graph.num_nodes