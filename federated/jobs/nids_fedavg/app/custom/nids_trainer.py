import json
import os
import sys
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from torch_geometric.loader import LinkNeighborLoader

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.dxo import DXO, DataKind, from_shareable as dxo_from_shareable

from graphids_model import GraphIDS
from utils.dataloaders import SequentialDataset, collate_fn
from utils.trainers import find_threshold, validate
from nids_data_loader import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"


class NIDSTrainer(Executor):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "NF-CSE-CIC-IDS2018-v3",
        local_epochs: int = 2,
        batch_size: int = 16384,
        ae_batch_size: int = 64,
        window_size: int = 512,
        step_percent: float = 1.0,
        fanout: int = 32768,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.6,
        ae_weight_decay: float = 0.03986574357073468,
        edim_out: int = 64,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.5,
        ae_dropout: float = 0.2,
        agg_type: str = "mean",
        mask_ratio: float = 0.15,
        positional_encoding=None,
        client_id: str = None,
        checkpoint_dir: str = "checkpoints",
        early_stop_patience: int = 3,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.ae_batch_size = ae_batch_size
        self.window_size = window_size
        self.step_percent = step_percent
        self.fanout = fanout
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.ae_weight_decay = ae_weight_decay
        self.edim_out = edim_out
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.ae_dropout = ae_dropout
        self.agg_type = agg_type
        self.mask_ratio = mask_ratio
        self.positional_encoding = positional_encoding
        self.client_id = client_id
        self.checkpoint_dir = checkpoint_dir
        self.early_stop_patience = early_stop_patience

        # Initialized lazily on first round
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None

        # Best-model tracking and per-round metrics across rounds
        self.best_global_val_pr_auc = 0.0
        self.best_local_val_pr_auc = float("-inf")
        self.rounds_no_improve = 0
        self.early_stopped = False
        self.metrics_history: list = []

    # ------------------------------------------------------------------ #
    # NVFlare entry point                                                  #
    # ------------------------------------------------------------------ #
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name == "train":
            return self._train(shareable, fl_ctx, abort_signal)
        return make_reply(ReturnCode.TASK_UNKNOWN)

    # ------------------------------------------------------------------ #
    # Lazy init — runs once on the very first round                        #
    # ------------------------------------------------------------------ #
    def _lazy_init(self, fl_ctx: FLContext):
        if self.model is not None:
            return

        self.log_info(fl_ctx, "Initializing model and data loaders...")

        self.train_loader, self.val_loader, ndim_in, edim_in = get_loaders(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            fanout=self.fanout,
            client_id=self.client_id,
            shuffle=(self.positional_encoding is None),
        )

        self.model = GraphIDS(
            ndim_in=ndim_in,
            edim_in=edim_in,
            edim_out=self.edim_out,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=self.window_size,
            dropout=self.dropout,
            ae_dropout=self.ae_dropout,
            positional_encoding=self.positional_encoding,
            agg_type=self.agg_type,
            mask_ratio=self.mask_ratio,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "weight_decay": self.weight_decay},
                {"params": self.model.transformer.parameters(), "weight_decay": self.ae_weight_decay},
            ],
            lr=self.lr,
        )

        self.log_info(fl_ctx, f"Model ready — edim_in={edim_in}, device={device}")

    def _clone_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------ #
    # Training round                                                       #
    # ------------------------------------------------------------------ #
    def _train(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        self._lazy_init(fl_ctx)

        # --- 1. Pull global weights from server (arrive as numpy arrays) ---
        dxo = dxo_from_shareable(shareable)
        global_params = {
            k: torch.tensor(v, dtype=torch.float32).to(device)
            for k, v in dxo.data.items()
        }
        self.model.load_state_dict(global_params)

        # Snapshot before training for diff computation
        global_weights = {k: v.clone().cpu() for k, v in global_params.items()}

        current_round = dxo.get_meta_prop("current_round", 0)
        site_name = fl_ctx.get_identity_name()

        # --- 2. Validate global model and save best checkpoint ---
        _, val_errors, val_labels = validate(
            self.model, self.val_loader, self.ae_batch_size, self.window_size, device
        )
        # Macro PR-AUC for binary classification: average of positive and negative class PR-AUCs
        pr_auc_pos = average_precision_score(val_labels.cpu(), val_errors.cpu())
        pr_auc_neg = average_precision_score(1 - val_labels.cpu(), 1 - val_errors.cpu())
        val_pr_auc = (pr_auc_pos + pr_auc_neg) / 2
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        val_pred = (val_errors > threshold).int()
        val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average="macro", zero_division=0)
        self.log_info(
            fl_ctx,
            f"Round {current_round} — global model: val_pr_auc={val_pr_auc:.4f}, "
            f"val_macro_f1={val_f1:.4f}, threshold={threshold:.6f}",
        )

        if val_pr_auc >= self.best_global_val_pr_auc:
            self.best_global_val_pr_auc = val_pr_auc
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoint_dir, f"best_global_model_{site_name}.pt")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "round": current_round,
                    "val_pr_auc": val_pr_auc,
                    "val_macro_f1": val_f1,
                    "threshold": threshold,
                },
                ckpt_path,
            )
            self.log_info(fl_ctx, f"  New best model saved → {ckpt_path}")

        if self.early_stopped:
            self.log_info(
                fl_ctx,
                f"Round {current_round} — early-stopped; skipping local training and sending zero diff",
            )
            zero_diff = {k: torch.zeros_like(v).cpu().numpy() for k, v in global_weights.items()}
            out_dxo = DXO(
                data_kind=DataKind.WEIGHT_DIFF,
                data=zero_diff,
                meta={
                    "num_steps_current_round": 0,
                    "early_stopped": True,
                    "rounds_no_improve": self.rounds_no_improve,
                },
            )
            return out_dxo.to_shareable()

        # --- 3. Local training ---
        self.log_info(
            fl_ctx,
            f"Round {current_round} — {self.local_epochs} local epoch(s) on {device}",
        )

        total_train_loss = 0.0
        best_round_state = None
        best_round_epoch = None
        best_round_metrics = None

        for epoch in range(self.local_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            epoch_loss = self._run_epoch(abort_signal)
            total_train_loss += epoch_loss

            val_loss_local, val_errors_local, val_labels_local = validate(
                self.model, self.val_loader, self.ae_batch_size, self.window_size, device
            )
            # Macro PR-AUC for binary classification: average of positive and negative class PR-AUCs
            pr_auc_pos_local = average_precision_score(val_labels_local.cpu(), val_errors_local.cpu())
            pr_auc_neg_local = average_precision_score(1 - val_labels_local.cpu(), 1 - val_errors_local.cpu())
            val_pr_auc_local = (pr_auc_pos_local + pr_auc_neg_local) / 2
            threshold_local = find_threshold(val_errors_local, val_labels_local, method="supervised")
            val_pred_local = (val_errors_local > threshold_local).int()
            val_f1_local = f1_score(
                val_labels_local.cpu(), val_pred_local.cpu(), average="macro", zero_division=0
            )

            self.log_info(
                fl_ctx,
                f"  Epoch {epoch + 1}/{self.local_epochs} — loss: {epoch_loss:.6f}, "
                f"val_loss: {val_loss_local:.6f}, val_pr_auc: {val_pr_auc_local:.4f}, "
                f"val_macro_f1: {val_f1_local:.4f}",
            )

            if best_round_metrics is None or val_pr_auc_local > best_round_metrics["val_pr_auc"]:
                best_round_state = self._clone_state_dict()
                best_round_epoch = epoch + 1
                best_round_metrics = {
                    "val_loss": val_loss_local,
                    "val_pr_auc": val_pr_auc_local,
                    "val_macro_f1": val_f1_local,
                    "threshold": threshold_local,
                }

        total_train_loss /= max(self.local_epochs, 1)

        if best_round_state is None:
            best_round_state = self._clone_state_dict()
            best_round_epoch = self.local_epochs
            best_round_metrics = {
                "val_loss": 0.0,
                "val_pr_auc": float("-inf"),
                "val_macro_f1": 0.0,
                "threshold": 0.0,
            }

        separator = "-" * 80
        self.log_info(fl_ctx, separator)
        self.log_info(
            fl_ctx,
            f"Round {current_round} complete — "
            f"train_loss={total_train_loss:.6f}, val_loss={best_round_metrics['val_loss']:.6f}, "
            f"val_pr_auc={best_round_metrics['val_pr_auc']:.4f}, "
            f"val_macro_f1={best_round_metrics['val_macro_f1']:.4f}",
        )
        self.log_info(fl_ctx, separator)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        local_ckpt_path = os.path.join(
            self.checkpoint_dir, f"best_local_model_{site_name}_round_{current_round}.pt"
        )
        torch.save(
            {
                "model_state_dict": best_round_state,
                "round": current_round,
                "epoch": best_round_epoch,
                "val_pr_auc": best_round_metrics["val_pr_auc"],
                "val_macro_f1": best_round_metrics["val_macro_f1"],
                "val_loss": best_round_metrics["val_loss"],
                "threshold": best_round_metrics["threshold"],
            },
            local_ckpt_path,
        )

        # Persist per-round metrics for post-training reporting
        self.metrics_history.append(
            {
                "round": current_round,
                "train_loss": total_train_loss,
                "val_loss": best_round_metrics["val_loss"],
                "val_pr_auc": best_round_metrics["val_pr_auc"],
                "val_macro_f1": best_round_metrics["val_macro_f1"],
                "best_epoch": best_round_epoch,
            }
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_history_{site_name}.json")
        with open(metrics_path, "w") as _f:
            json.dump(self.metrics_history, _f, indent=2)

        if best_round_metrics["val_pr_auc"] > self.best_local_val_pr_auc:
            self.best_local_val_pr_auc = best_round_metrics["val_pr_auc"]
            self.rounds_no_improve = 0
        else:
            self.rounds_no_improve += 1
            if self.rounds_no_improve >= self.early_stop_patience:
                self.early_stopped = True
                self.log_info(
                    fl_ctx,
                    f"Early stop triggered after {self.rounds_no_improve} rounds without improvement",
                )

        # --- 4. Compute weight diff and convert to numpy for DXO ---
        if self.early_stopped:
            weight_diff = {k: torch.zeros_like(v).cpu().numpy() for k, v in global_weights.items()}
        else:
            weight_diff = {
                k: (best_round_state[k] - global_weights[k]).numpy()
                for k in global_weights
            }

        # --- 5. Return as DXO (stable across all NVFlare versions) ---
        num_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else 1
        out_dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=weight_diff,
            meta={
                "num_steps_current_round": num_samples,
                "early_stopped": self.early_stopped,
                "rounds_no_improve": self.rounds_no_improve,
            },
        )
        return out_dxo.to_shareable()

    # ------------------------------------------------------------------ #
    # One full epoch — mirrors centralized train_encoder exactly           #
    # ------------------------------------------------------------------ #
    def _run_epoch(self, abort_signal: Signal) -> float:
        self.model.train()
        criterion = nn.MSELoss(reduction="none")
        step = int(self.window_size * self.step_percent)
        total_loss = 0.0

        for batch in self.train_loader:
            if abort_signal.triggered:
                break

            batch.batch_edge_couples = batch.edge_label_index.t()
            batch = batch.to(device)

            # GNN forward → edge embeddings
            train_emb = self.model.encoder(
                batch.edge_index,
                batch.edge_attr,
                batch.batch_edge_couples,
                batch.num_nodes,
            )

            # Sliding window AE loader (identical to centralized)
            ae_loader = DataLoader(
                SequentialDataset(
                    train_emb,
                    window=self.window_size,
                    step=step,
                    device=device,
                ),
                batch_size=self.ae_batch_size,
                collate_fn=collate_fn,
            )

            # Accumulate loss, single backward per GNN batch
            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0

            for ae_batch, mask in ae_loader:
                outputs = self.model.transformer(ae_batch, mask)
                # No detach on target — matches centralized exactly
                loss = criterion(outputs, ae_batch)
                loss = torch.sum(loss * mask) / torch.sum(mask)
                accumulated_loss += loss
                seq_count += 1

            if seq_count > 0:
                mean_loss = accumulated_loss / seq_count
                total_loss += mean_loss.item()
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / max(len(self.train_loader), 1)