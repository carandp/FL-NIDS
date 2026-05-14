import json
import math
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
        oversample_min_ratio: float = 0.05,
        oversample_target_ratio: float = 0.3,
        oversample_method: str = "borderline-1",
        oversample_random_state: int = 42,
        dp_enabled: bool = True,
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 1.0,
        dp_delta: float = 1e-5,
        dp_accountant_orders=None,
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
        self.oversample_min_ratio = oversample_min_ratio
        self.oversample_target_ratio = oversample_target_ratio
        self.oversample_method = oversample_method
        self.oversample_random_state = oversample_random_state
        self.dp_enabled = dp_enabled
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_delta = dp_delta
        self.dp_accountant_orders = dp_accountant_orders

        # Initialized lazily on first round
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None

        self.dp_steps_total = 0
        self.dp_dataset_size = None
        self.dp_batch_size = None

        # Best-model tracking and per-round metrics across rounds
        self.best_val_pr_auc = 0.0
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
        
        # 1. Loaders FIRST — dp_dataset_size depends on this
        self.train_loader, self.val_loader, ndim_in, edim_in = get_loaders(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            batch_size=self.batch_size,
            fanout=self.fanout,
            client_id=self.client_id,
            shuffle=(self.positional_encoding is None),
            oversample_min_ratio=self.oversample_min_ratio,
            oversample_target_ratio=self.oversample_target_ratio,
            oversample_method=self.oversample_method,
            oversample_random_state=self.oversample_random_state,
        )

        # 2. Model
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

        # 3. Optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "weight_decay": self.weight_decay},
                {"params": self.model.transformer.parameters(), "weight_decay": self.ae_weight_decay},
            ],
            lr=self.lr,
        )

        # 4. DP setup — AFTER loaders exist so dataset size is known
        if self.dp_enabled:
            # Assign dataset size and raw batch size from the loader
            self.dp_dataset_size = (
                len(self.train_loader.dataset)
                if hasattr(self.train_loader, "dataset")
                else self.batch_size
            )
            self.dp_batch_size = getattr(self.train_loader, "batch_size", self.batch_size)

            # Cap sample rate at 5% to prevent ε → ∞ on small datasets
            MAX_SAMPLE_RATE = 0.05
            max_safe_bs = max(1, int(self.dp_dataset_size * MAX_SAMPLE_RATE))
            effective_bs = min(self.dp_batch_size, self.dp_dataset_size, max_safe_bs)
            self.dp_batch_size = effective_bs

            # Adapt window size so the AE inner loader is never empty.
            # Rule: window ≤ 10% of expected embeddings per batch.
            safe_window = max(8, min(self.window_size, effective_bs // 10))
            if safe_window < self.window_size:
                self.log_info(
                    fl_ctx,
                    f"window_size reduced {self.window_size} → {safe_window} "
                    f"(dataset={self.dp_dataset_size}, effective_bs={effective_bs})",
                )
                self.window_size = safe_window

            self.log_info(fl_ctx, (
                f"DP config — dataset={self.dp_dataset_size}, "
                f"effective_bs={self.dp_batch_size}, "
                f"sample_rate={self.dp_batch_size / self.dp_dataset_size:.4f}, "
                f"window={self.window_size}"
            ))

        self.log_info(fl_ctx, f"Model ready — edim_in={edim_in}, device={device}")

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

        # --- 2. Validate global model and save best checkpoint ---
        _, val_errors, val_labels = validate(
            self.model, self.val_loader, self.ae_batch_size, self.window_size, device
        )
        pr_auc_pos = average_precision_score(val_labels.cpu(), val_errors.cpu())
        pr_auc_neg = average_precision_score(1 - val_labels.cpu(), 1 - val_errors.cpu())
        val_pr_auc = (pr_auc_pos + pr_auc_neg) / 2
        threshold = find_threshold(val_errors, val_labels, method="supervised")
        val_pred = (val_errors > threshold).int()
        val_f1 = f1_score(val_labels.cpu(), val_pred.cpu(), average="macro", zero_division=0)
        site_name = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Round {current_round} — global model: val_pr_auc={val_pr_auc:.4f}, "
            f"val_macro_f1={val_f1:.4f}, threshold={threshold:.6f}",
        )

        if val_pr_auc >= self.best_val_pr_auc:
            self.best_val_pr_auc = val_pr_auc
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

        # --- 3. Local training ---
        self.log_info(
            fl_ctx,
            f"Round {current_round} — {self.local_epochs} local epoch(s) on {device}",
        )

        total_train_loss = 0.0
        dp_step_count = 0
        dp_grad_norm_sum = 0.0
        dp_clip_coef_sum = 0.0

        for epoch in range(self.local_epochs):
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            epoch_loss, epoch_dp_steps, epoch_grad_norm, epoch_clip_coef = self._run_epoch(
                abort_signal
            )
            total_train_loss += epoch_loss
            dp_step_count += epoch_dp_steps
            dp_grad_norm_sum += epoch_grad_norm
            dp_clip_coef_sum += epoch_clip_coef
            self.log_info(
                fl_ctx,
                f"  Epoch {epoch + 1}/{self.local_epochs} — loss: {epoch_loss:.6f}",
            )
        total_train_loss /= max(self.local_epochs, 1)

        # --- 4. DP accounting ---
        if self.dp_enabled:
            self.dp_steps_total += dp_step_count

            dp_sample_rate = (
                float(self.dp_batch_size) / float(self.dp_dataset_size)
                if self.dp_dataset_size
                else 1.0
            )
            dp_sample_rate = max(0.0, min(dp_sample_rate, 1.0))

            # Keep epsilon as a raw float throughout — only stringify for JSON
            dp_epsilon_raw, dp_best_order = self._compute_privacy_epsilon(
                steps=self.dp_steps_total,
                sample_rate=dp_sample_rate,
            )

            if not math.isfinite(dp_epsilon_raw):
                self.log_warning(fl_ctx, (
                    f"DP epsilon is non-finite ({dp_epsilon_raw}) at "
                    f"step={self.dp_steps_total}, "
                    f"sample_rate={dp_sample_rate:.6f}, "
                    f"noise_multiplier={self.dp_noise_multiplier}. "
                    f"Dataset may be too small for the current batch size."
                ))

            dp_epsilon = dp_epsilon_raw  # float (may be inf)
            dp_epsilon_json = dp_epsilon if math.isfinite(dp_epsilon) else "inf"  # JSON-safe

            dp_grad_norm_mean = dp_grad_norm_sum / max(dp_step_count, 1)
            dp_clip_coef_mean = dp_clip_coef_sum / max(dp_step_count, 1)

            eps_display = f"{dp_epsilon:.4f}" if math.isfinite(dp_epsilon) else "inf"
            self.log_info(fl_ctx, (
                f"DP-SGD: steps={self.dp_steps_total}, eps={eps_display}, "
                f"delta={self.dp_delta:.2e}, noise={self.dp_noise_multiplier:.4f}, "
                f"clip={self.dp_clip_norm:.4f}, sample_rate={dp_sample_rate:.6f}"
            ))
        else:
            dp_epsilon_json = None
            dp_best_order = None
            dp_sample_rate = None
            dp_grad_norm_mean = None
            dp_clip_coef_mean = None

        # --- 5. Validate locally-trained model ---
        val_loss_local, val_errors_local, val_labels_local = validate(
            self.model, self.val_loader, self.ae_batch_size, self.window_size, device
        )
        pr_auc_pos_local = average_precision_score(val_labels_local.cpu(), val_errors_local.cpu())
        pr_auc_neg_local = average_precision_score(1 - val_labels_local.cpu(), 1 - val_errors_local.cpu())
        val_pr_auc_local = (pr_auc_pos_local + pr_auc_neg_local) / 2
        threshold_local = find_threshold(val_errors_local, val_labels_local, method="supervised")
        val_pred_local = (val_errors_local > threshold_local).int()
        val_f1_local = f1_score(val_labels_local.cpu(), val_pred_local.cpu(), average="macro", zero_division=0)

        separator = "-" * 80
        self.log_info(fl_ctx, separator)
        self.log_info(fl_ctx, (
            f"Round {current_round} complete — "
            f"train_loss={total_train_loss:.6f}, val_loss={val_loss_local:.6f}, "
            f"val_pr_auc={val_pr_auc_local:.4f}, val_macro_f1={val_f1_local:.4f}"
        ))
        self.log_info(fl_ctx, separator)

        # --- 6. Persist per-round metrics ---
        self.metrics_history.append({
            "round": current_round,
            "train_loss": total_train_loss,
            "val_loss": val_loss_local,
            "val_pr_auc": val_pr_auc_local,
            "val_macro_f1": val_f1_local,
            "dp_enabled": self.dp_enabled,
            "dp_epsilon": dp_epsilon_json,
            "dp_delta": self.dp_delta if self.dp_enabled else None,
            "dp_noise_multiplier": self.dp_noise_multiplier if self.dp_enabled else None,
            "dp_clip_norm": self.dp_clip_norm if self.dp_enabled else None,
            "dp_sample_rate": dp_sample_rate,
            "dp_steps_total": self.dp_steps_total if self.dp_enabled else None,
            "dp_steps_round": dp_step_count if self.dp_enabled else None,
            "dp_best_order": dp_best_order,
            "dp_grad_norm": dp_grad_norm_mean,
            "dp_clip_coef": dp_clip_coef_mean,
        })
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        metrics_path = os.path.join(self.checkpoint_dir, f"metrics_history_{site_name}.json")
        with open(metrics_path, "w") as _f:
            json.dump(self.metrics_history, _f, indent=2)

        # --- 7. Compute weight diff and convert to numpy for DXO ---
        local_weights = self.model.state_dict()
        weight_diff = {
            k: (local_weights[k].cpu() - global_weights[k]).numpy()
            for k in local_weights
        }

        # --- 8. Return as DXO ---
        num_samples = (
            len(self.train_loader.dataset)
            if hasattr(self.train_loader, "dataset")
            else 1
        )
        out_dxo = DXO(
            data_kind=DataKind.WEIGHT_DIFF,
            data=weight_diff,
            meta={"num_steps_current_round": num_samples},
        )
        return out_dxo.to_shareable()

    # ------------------------------------------------------------------ #
    # One full epoch                                                        #
    # ------------------------------------------------------------------ #
    def _run_epoch(self, abort_signal: Signal) -> tuple[float, int, float, float]:
        self.model.train()
        criterion = nn.MSELoss(reduction="none")
        step = int(self.window_size * self.step_percent)
        total_loss = 0.0
        dp_step_count = 0
        dp_grad_norm_sum = 0.0
        dp_clip_coef_sum = 0.0

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

            accumulated_loss = torch.tensor(0.0, device=device)
            seq_count = 0

            if train_emb.shape[0] < self.window_size:
                # Too few embeddings for windowed AE — train on full sequence directly
                outputs = self.model.transformer(train_emb.unsqueeze(0), None)
                loss = nn.MSELoss()(outputs.squeeze(0), train_emb)
                accumulated_loss += loss
                seq_count += 1
            else:
                # Normal windowed AE path
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
                for ae_batch, mask in ae_loader:
                    outputs = self.model.transformer(ae_batch, mask)
                    loss = criterion(outputs, ae_batch)
                    loss = torch.sum(loss * mask) / torch.sum(mask)
                    accumulated_loss += loss
                    seq_count += 1

            if seq_count > 0:
                mean_loss = accumulated_loss / seq_count
                total_loss += mean_loss.item()
                mean_loss.backward()
                if self.dp_enabled:
                    grad_norm, clip_coef = self._dp_clip_and_add_noise()
                    dp_step_count += 1
                    dp_grad_norm_sum += grad_norm
                    dp_clip_coef_sum += clip_coef
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return (
            total_loss / max(len(self.train_loader), 1),
            dp_step_count,
            dp_grad_norm_sum,
            dp_clip_coef_sum,
        )

    # ------------------------------------------------------------------ #
    # DP: gradient clipping + noise injection                              #
    # ------------------------------------------------------------------ #
    def _dp_clip_and_add_noise(self) -> tuple[float, float]:
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return 0.0, 1.0

        total_norm = torch.norm(
            torch.stack([p.grad.detach().norm(2) for p in params]), 2
        )
        clip_coef = float(self.dp_clip_norm / (total_norm + 1e-6))
        if clip_coef < 1.0:
            for p in params:
                p.grad.mul_(clip_coef)

        if self.dp_noise_multiplier > 0:
            batch_divisor = max(float(self.dp_batch_size or 1), 1.0)
            noise_std = self.dp_noise_multiplier * self.dp_clip_norm / batch_divisor
            for p in params:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=p.grad.shape,
                    device=p.grad.device,
                )
                p.grad.add_(noise)

        return float(total_norm.item()), min(clip_coef, 1.0)

    # ------------------------------------------------------------------ #
    # DP: RDP-based epsilon accounting                                     #
    # ------------------------------------------------------------------ #
    def _compute_privacy_epsilon(self, steps: int, sample_rate: float) -> tuple[float, float | None]:
        if self.dp_noise_multiplier <= 0 or steps <= 0 or sample_rate <= 0:
            return 0.0, None

        orders = self.dp_accountant_orders
        if orders is None:
            orders = [1.25, 1.5, 2, 3, 5, 10, 20, 50, 100]

        rdp = []
        for order in orders:
            rdp.append(
                steps * (sample_rate ** 2) * order / (2 * (self.dp_noise_multiplier ** 2))
            )

        epsilons = [
            rdp_i + math.log(1.0 / self.dp_delta) / (order - 1)
            for rdp_i, order in zip(rdp, orders)
        ]
        min_idx = int(min(range(len(epsilons)), key=epsilons.__getitem__))
        return float(epsilons[min_idx]), float(orders[min_idx])