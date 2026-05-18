"""
RDP Accountant for the Gaussian mechanism with Poisson subsampling.

References:
  Mironov 2017  – Rényi Differential Privacy
  Wang et al. 2019 – Subsampled Rényi DP
  Balle et al. 2020 – tight RDP→(ε,δ)-DP conversion
"""
from __future__ import annotations

import numpy as np
from math import lgamma, log, exp, inf
from typing import List


# Fine-grained order grid: fractional orders catch tight optima during conversion.
_ORDERS: List[float] = (
    [1 + k / 10.0 for k in range(1, 100)]   # 1.1 … 10.9
    + list(range(11, 65))                     # 11 … 64
    + [128, 256, 512]
)


def _log_comb(n: int, k: int) -> float:
    """log C(n, k) using lgamma for large values."""
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP of Gaussian mechanism (sensitivity 1, std sigma): alpha / (2 sigma^2)."""
    return alpha / (2.0 * sigma ** 2)


def _rdp_subsampled_integer(alpha: int, sigma: float, q: float) -> float:
    """
    Exact RDP at integer order alpha for Poisson-subsampled Gaussian.

    From Wang et al. 2019, Proposition 3:
        RDP_subsampled(alpha) = 1/(alpha-1) * log(
            sum_{k=0}^{alpha} C(alpha,k) * q^k * (1-q)^(alpha-k) * exp(k(k-1)/(2sigma^2))
        )

    The k(k-1)/(2sigma^2) term is the log-MGF of the Gaussian mechanism
    at order k, i.e. log E[e^{(k-1) Z / sigma^2}] for Z~N(0,1).
    """
    if alpha == 1:
        # L'Hôpital limit: q * RDP_gaussian(2, sigma)
        return q * _rdp_gaussian(2.0, sigma)

    log_terms = []
    for k in range(0, alpha + 1):
        log_binom   = _log_comb(alpha, k)
        log_q_term  = k * log(q) + (alpha - k) * log(1.0 - q)
        log_moment  = k * (k - 1) / (2.0 * sigma ** 2)   # = log E[e^{k(k-1)Z/sigma^2}]
        log_terms.append(log_binom + log_q_term + log_moment)

    # Numerically stable log-sum-exp
    max_lt = max(log_terms)
    log_sum = max_lt + log(sum(exp(lt - max_lt) for lt in log_terms))
    return log_sum / (alpha - 1)


def _rdp_one_step(sigma: float, q: float) -> np.ndarray:
    """
    Full RDP curve (one array of len(ORDERS)) for a single application of the
    Poisson-subsampled Gaussian mechanism.
    """
    if q == 0.0:
        return np.zeros(len(_ORDERS))
    if q == 1.0:
        return np.array([_rdp_gaussian(a, sigma) for a in _ORDERS])

    rdp = np.zeros(len(_ORDERS))
    for i, alpha in enumerate(_ORDERS):
        # Use ceiling to get a valid integer order; the exact formula only
        # holds at integers, so we over-estimate by rounding up.
        rdp[i] = _rdp_subsampled_integer(max(2, int(np.ceil(alpha))), sigma, q)
    return rdp


def _rdp_to_dp(rdp_curve: np.ndarray, delta: float) -> float:
    """
    Convert accumulated RDP curve to (ε, δ)-DP.

    Uses the tight conversion from Balle et al. 2020 (Theorem 21):
        ε = RDP(alpha) + log((alpha-1)/alpha) - (log(delta) + log(alpha)) / (alpha-1)
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    best_eps = inf
    for alpha, rdp_val in zip(_ORDERS, rdp_curve):
        if rdp_val == 0.0 or alpha <= 1.0:
            continue
        try:
            eps = (rdp_val
                   + log((alpha - 1) / alpha)
                   - (log(delta) + log(alpha)) / (alpha - 1))
            if np.isfinite(eps) and eps < best_eps:
                best_eps = float(eps)
        except (ValueError, ZeroDivisionError):
            continue

    return best_eps


class RDPAccountant:
    """
    Tracks Rényi-DP privacy budget for one entity (one FL client).

    Usage
    -----
    acc = RDPAccountant()
    acc.step(noise_multiplier=1.1, clip_norm=1.0, sample_rate=0.003, num_steps=686)
    eps = acc.get_epsilon(delta=1e-5)
    """

    def __init__(self) -> None:
        self._rdp: np.ndarray = np.zeros(len(_ORDERS))
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        noise_multiplier: float,
        sample_rate: float,
        clip_norm: float,
        num_steps: int = 1
    ) -> None:
        """
        Account for `num_steps` applications of the (q, sigma) Gaussian mechanism.

        Parameters
        ----------
        noise_multiplier : float
            Ratio sigma / clip_norm (the "σ" in the DP literature).
        clip_norm : float
            L2 clipping bound C.
        sample_rate : float
            Poisson sampling probability q = batch_size / dataset_size.
        num_steps : int
            Number of gradient steps (batches) in this accounting period.
        """
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        sigma_actual = noise_multiplier * clip_norm
        self._rdp += num_steps * _rdp_one_step(sigma_actual, sample_rate)
        self._total_steps += num_steps

    def get_epsilon(self, delta: float = 1e-5) -> float:
        """Return current ε for the given δ."""
        return _rdp_to_dp(self._rdp, delta)

    @property
    def total_steps(self) -> int:
        return self._total_steps

    # ------------------------------------------------------------------
    # Serialisation (for persistence across rounds)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "rdp_curve": self._rdp.tolist(),
            "total_steps": self._total_steps,
            "num_orders": len(_ORDERS),
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "RDPAccountant":
        acc = cls()
        if d.get("num_orders") == len(_ORDERS):
            acc._rdp = np.array(d["rdp_curve"], dtype=float)
        # else: order grid changed – start fresh (conservative)
        acc._total_steps = d.get("total_steps", 0)
        return acc
