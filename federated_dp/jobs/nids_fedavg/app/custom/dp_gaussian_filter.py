"""
DPGaussianFilter – NVFlare task_result_filters plugin.

What it does
------------
1. Receives the client's outgoing Shareable (WEIGHT_DIFF).
2. Flattens all weight delta tensors into a single vector.
3. L2-clips the vector to `clip_norm`.
4. Adds isotropic Gaussian noise  N(0, (noise_multiplier * clip_norm)^2 I).
5. Reconstructs the original tensor structure.
6. Accounts for the privacy cost of this round and saves privacy history to disk.
7. Returns an empty Shareable if the privacy budget is exhausted.

Configuration (per client in config_fed_client.json)
-----------------------------------------------------
{
  "id": "dp_gaussian_filter",
  "path": "dp_gaussian_filter.DPGaussianFilter",
  "args": {
    "clip_norm":        1.0,
    "noise_multiplier": 1.1,
    "target_epsilon":   10.0,
    "target_delta":     1e-5,
        "dataset_size":     770415,
        "batch_size":       32,
        "local_epochs":     2
  }
}

Sampling-rate note
------------------
The RDP accountant uses q = batch_size / dataset_size to compute privacy cost.
Clients with smaller datasets (higher q) accumulate budget faster — make sure
each client sets *its own* dataset_size in the config.

Noise calibration
-----------------
For a desired (ε, δ) budget over T=100 rounds, use dp_noise_calibrator.py
(in this folder) to back-solve noise_multiplier before deploying.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ReturnCode

from budget_tracker import BudgetTracker

logger = logging.getLogger(__name__)


# Client-specific dataset sizes for privacy accounting.
_CLIENT_DATASET_SIZES = {
    "client0": 770415,
    "client1": 1394658,
    "client2": 10960,
}


class DPGaussianFilter(Filter):
    """
    Differential-privacy output filter for NVFlare clients.

    Place in task_result_filters so it intercepts WEIGHT_DIFF tensors
    *after* local training, *before* transmission to the server.
    """

    def __init__(
        self,
        clip_norm:        float = 1.0,
        noise_multiplier: float = 1.1,
        target_epsilon:   float = 10.0,
        target_delta:     float = 1e-5,
        dataset_size:     int   = 10_000,
        batch_size:       int   = 32,
        local_epochs:     int   = 2,
        checkpoint_dir:   str   = "checkpoints",
    ) -> None:
        super().__init__()

        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not (0 < target_delta < 1):
            raise ValueError("target_delta must be in (0, 1)")

        self.clip_norm        = clip_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon   = target_epsilon
        self.target_delta     = target_delta
        self.dataset_size     = dataset_size
        self.batch_size       = batch_size
        self.local_epochs     = local_epochs
        self.checkpoint_dir   = checkpoint_dir

        # Tracker is created lazily (needs client_id from fl_ctx)
        self._tracker: Optional[BudgetTracker] = None
        self._privacy_history: list = []

    # ------------------------------------------------------------------
    # NVFlare Filter interface
    # ------------------------------------------------------------------

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        client_id = fl_ctx.get_identity_name()
        current_round = None

        tracker = self._get_tracker(client_id)

        # ── Guard: skip if budget already exhausted from a previous round ──
        if tracker.is_exhausted():
            logger.warning(
                "[DP/%s] Budget exhausted before round %d. Returning empty update.",
                client_id, current_round,
            )
            return self._empty_shareable()

        # ── Extract WEIGHT_DIFF DXO ──
        try:
            dxo = from_shareable(shareable)
        except Exception as exc:
            logger.error("[DP/%s] Could not extract DXO: %s", client_id, exc)
            shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
            return shareable

        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            # Pass through untouched for any other data kind
            return shareable

        weight_diff: dict = dxo.data

        # ── Step 1: clip ──
        flat, shapes, keys = self._flatten(weight_diff)
        original_norm = float(np.linalg.norm(flat))
        if original_norm > self.clip_norm:
            flat = flat * (self.clip_norm / original_norm)
        clipped_norm = float(np.linalg.norm(flat))

        # ── Step 2: add Gaussian noise ──
        noise_std = tracker.noise_std()
        flat = flat + np.random.normal(0.0, noise_std, flat.shape).astype(flat.dtype)

        noisy_norm = float(np.linalg.norm(flat))
        logger.debug(
            "[DP/%s] round=%d  ||Δw||=%.4f  clipped=%.4f  noisy=%.4f  noise_std=%.4f",
            client_id, current_round, original_norm, clipped_norm, noisy_norm, noise_std,
        )

        # ── Step 3: reconstruct ──
        dxo.data = self._unflatten(flat, shapes, keys)

        # ── Step 4: account for privacy cost this round ──
        eps, current_round = tracker.account_round(current_round)

        # Attach ε metadata so the server/researcher can log it
        dxo.set_meta_prop("dp_epsilon",          round(eps, 6))
        dxo.set_meta_prop("dp_delta",            self.target_delta)
        dxo.set_meta_prop("dp_noise_std",        round(noise_std, 6))
        dxo.set_meta_prop("dp_clip_norm",        self.clip_norm)
        dxo.set_meta_prop("dp_original_norm",    round(original_norm, 6))
        dxo.set_meta_prop("dp_budget_remaining", round(tracker.remaining_budget(), 6))

        # Persist per-round privacy metrics (mirrors nids_trainer history style)
        self._privacy_history.append(
            {
                "round": current_round,
                "epsilon": round(eps, 6),
                "delta": self.target_delta,
                "clip_norm": self.clip_norm,
                "noise_multiplier": self.noise_multiplier,
                "budget_exhausted": tracker.is_exhausted(),
            }
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        history_path = os.path.join(self.checkpoint_dir, f"privacy_history_{client_id}.json")
        with open(history_path, "w") as history_file:
            json.dump(self._privacy_history, history_file, indent=2)

        # ── Step 5: abort if this round just exhausted the budget ──
        if tracker.is_exhausted():
            logger.warning(
                "[DP/%s] Budget just exhausted at round %d (ε=%.4f). "
                "Sending empty update for this round.",
                client_id, current_round, eps,
            )
            return self._empty_shareable()

        return dxo.to_shareable()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tracker(self, client_id: str) -> BudgetTracker:
        if self._tracker is None:
            dataset_size = _CLIENT_DATASET_SIZES.get(client_id, self.dataset_size)
            if dataset_size != self.dataset_size:
                logger.info(
                    "[DP/%s] Using dataset_size=%d from client map (config=%d)",
                    client_id, dataset_size, self.dataset_size,
                )
            elif client_id not in _CLIENT_DATASET_SIZES:
                logger.warning(
                    "[DP/%s] No dataset_size override found; using config value=%d",
                    client_id, self.dataset_size,
                )
            budget_file = os.path.join(self.checkpoint_dir, f"dp_budget_{client_id}.json")
            self._tracker = BudgetTracker(
                client_id=client_id,
                budget_file=budget_file,
                noise_multiplier=self.noise_multiplier,
                clip_norm=self.clip_norm,
                dataset_size=dataset_size,
                batch_size=self.batch_size,
                local_epochs=self.local_epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
            )
        return self._tracker

    @staticmethod
    def _flatten(weight_diff: dict):
        keys  = list(weight_diff.keys())
        parts = []
        shapes = {}
        for k in keys:
            arr = np.array(weight_diff[k], dtype=np.float32)
            shapes[k] = arr.shape
            parts.append(arr.ravel())
        flat = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return flat, shapes, keys

    @staticmethod
    def _unflatten(flat: np.ndarray, shapes: dict, keys: list) -> dict:
        result = {}
        idx = 0
        for k in keys:
            size = int(np.prod(shapes[k]))
            result[k] = flat[idx : idx + size].reshape(shapes[k])
            idx += size
        return result

    @staticmethod
    def _empty_shareable() -> Shareable:
        """Return a Shareable that the server will recognise but ignore gracefully."""
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data={})
        dxo.set_meta_prop("dp_budget_exhausted", True)
        s = dxo.to_shareable()
        s.set_return_code(ReturnCode.OK)
        return s
