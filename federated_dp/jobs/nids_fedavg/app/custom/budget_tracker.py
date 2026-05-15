"""
Budget tracker: wraps RDPAccountant with JSON persistence and early-stop logic.

Each FL client owns exactly one BudgetTracker instance.  The tracker is
created inside DPGaussianFilter.__init__ and survives across rounds by
reading/writing a JSON file on the client's local filesystem.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from rdp_accountant import RDPAccountant

logger = logging.getLogger(__name__)


@dataclass
class BudgetSnapshot:
    """What we store on disk after every round."""
    client_id: str
    round_num: int
    total_steps: int
    current_epsilon: float
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    clip_norm: float
    dataset_size: int
    batch_size: int
    local_epochs: int
    budget_exhausted: bool
    history: list = field(default_factory=list)  # list of (round, eps) pairs


class BudgetTracker:
    """
    Manages one RDPAccountant for a single FL client.

    Parameters
    ----------
    client_id : str
        Unique client identifier (used in log messages and the JSON filename).
    budget_file : str
        Absolute path where the tracker state is persisted.
    noise_multiplier : float
        σ / clip_norm used in the DP filter (must match!).
    clip_norm : float
        L2 clipping bound C.
    dataset_size : int
        Number of training rows on this client.
    batch_size : int
        Mini-batch size used during local training.
    local_epochs : int
        Number of local epochs per FL round.
    target_epsilon : float
        Maximum allowed privacy loss (training halts if this is exceeded).
    target_delta : float
        Target δ for (ε, δ)-DP.
    """

    def __init__(
        self,
        client_id: str,
        budget_file: str,
        noise_multiplier: float,
        clip_norm: float,
        dataset_size: int,
        batch_size: int,
        local_epochs: int,
        target_epsilon: float = 10.0,
        target_delta: float = 1e-5,
    ) -> None:
        self.client_id      = client_id
        self.budget_file    = budget_file
        self.noise_multiplier = noise_multiplier
        self.clip_norm      = clip_norm
        self.dataset_size   = dataset_size
        self.batch_size     = batch_size
        self.local_epochs   = local_epochs
        self.target_epsilon = target_epsilon
        self.target_delta   = target_delta

        # Derived quantities
        self.sample_rate    = batch_size / dataset_size
        self.steps_per_round = int(local_epochs * dataset_size / batch_size)

        # Load or create accountant
        self._accountant, self._snapshot = self._load_or_init()

    # ------------------------------------------------------------------
    # Core methods called by the DP filter
    # ------------------------------------------------------------------

    def account_round(self, current_round: Optional[int] = None) -> tuple[float, int]:
        """
        Step the accountant for one FL round and persist state.

        Returns
        -------
        (float, int)
            Current ε after this round, and the round number used.
        """
        if current_round is None:
            current_round = self._snapshot.round_num + 1
        self._accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
            num_steps=self.steps_per_round,
        )
        eps = self._accountant.get_epsilon(self.target_delta)
        exhausted = eps >= self.target_epsilon

        self._snapshot.round_num        = current_round
        self._snapshot.total_steps      = self._accountant.total_steps
        self._snapshot.current_epsilon  = eps
        self._snapshot.budget_exhausted = exhausted
        self._snapshot.history.append({"round": current_round, "epsilon": round(eps, 6)})

        self._save()

        logger.info(
            "[DP/%s] round=%d  ε=%.4f / %.1f  δ=%.0e  steps=%d  q=%.6f",
            self.client_id,
            current_round,
            eps,
            self.target_epsilon,
            self.target_delta,
            self._accountant.total_steps,
            self.sample_rate,
        )
        if exhausted:
            logger.warning(
                "[DP/%s] ⚠  Privacy budget EXHAUSTED at round %d (ε=%.4f > %.1f). "
                "Skipping weight update upload.",
                self.client_id,
                current_round,
                eps,
                self.target_epsilon,
            )
        return eps, current_round

    def is_exhausted(self) -> bool:
        return self._snapshot.budget_exhausted

    def remaining_budget(self) -> float:
        return max(0.0, self.target_epsilon - self._snapshot.current_epsilon)

    def noise_std(self) -> float:
        """Noise standard deviation to inject: σ_multiplier × clip_norm."""
        return self.noise_multiplier * self.clip_norm

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_or_init(self):
        if os.path.exists(self.budget_file):
            try:
                with open(self.budget_file) as f:
                    d = json.load(f)
                snap = BudgetSnapshot(**{k: v for k, v in d.items() if k != "_accountant_state"})
                acc = RDPAccountant.from_state_dict(d["_accountant_state"])
                logger.info(
                    "[DP/%s] Resumed from round %d  ε=%.4f",
                    self.client_id, snap.round_num, snap.current_epsilon,
                )
                return acc, snap
            except Exception as exc:
                logger.warning("[DP/%s] Could not load budget file (%s). Starting fresh.", self.client_id, exc)

        snap = BudgetSnapshot(
            client_id=self.client_id,
            round_num=0,
            total_steps=0,
            current_epsilon=0.0,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            noise_multiplier=self.noise_multiplier,
            clip_norm=self.clip_norm,
            dataset_size=self.dataset_size,
            batch_size=self.batch_size,
            local_epochs=self.local_epochs,
            budget_exhausted=False,
        )
        return RDPAccountant(), snap

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.budget_file) or ".", exist_ok=True)
        payload = asdict(self._snapshot)
        payload["_accountant_state"] = self._accountant.state_dict()
        with open(self.budget_file, "w") as f:
            json.dump(payload, f, indent=2)
