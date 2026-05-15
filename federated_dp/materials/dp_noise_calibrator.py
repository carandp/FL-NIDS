"""
dp_noise_calibrator.py
======================
Offline tool: given a target (ε, δ) and your FL setup, find the minimum
noise_multiplier that keeps all three clients within budget after 100 rounds.

Usage
-----
    python dp_noise_calibrator.py

Outputs a table like:

  σ=0.80  client_1  ε=28.41  client_2  ε=24.78  client_3  ε=52.13  ← over budget
  σ=1.00  client_1  ε=12.34  client_2  ε=11.02  client_3  ε=24.56  ← client_3 over
  σ=1.30  client_1  ε=7.21   client_2  ε=6.45   client_3  ε=9.87   ← all within 10.0 ✓

The binding client is always client_3 (smallest n, highest q).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List

from rdp_accountant import RDPAccountant


@dataclass
class ClientSpec:
    name: str
    dataset_size: int
    batch_size:   int = 32
    local_epochs: int = 2

    @property
    def sample_rate(self) -> float:
        return self.batch_size / self.dataset_size

    @property
    def steps_per_round(self) -> int:
        return int(self.local_epochs * self.dataset_size / self.batch_size)


def simulate_epsilon(
    spec: ClientSpec,
    noise_multiplier: float,
    num_rounds: int,
    delta: float,
) -> float:
    acc = RDPAccountant()
    for _ in range(num_rounds):
        acc.step(
            noise_multiplier=noise_multiplier,
            sample_rate=spec.sample_rate,
            num_steps=spec.steps_per_round,
        )
    return acc.get_epsilon(delta)


def calibrate(
    clients: List[ClientSpec],
    target_epsilon: float = 10.0,
    target_delta:   float = 1e-5,
    num_rounds:     int   = 100,
    sigma_range:    tuple = (0.5, 3.0),
    sigma_step:     float = 0.05,
    verbose:        bool  = True,
) -> float:
    """
    Binary-search for the smallest noise_multiplier where every client
    satisfies ε ≤ target_epsilon.

    Returns
    -------
    float
        Recommended noise_multiplier (or inf if no value in range works).
    """
    lo, hi = sigma_range
    best_sigma = float("inf")
    best_results = {}

    if verbose:
        header = f"{'σ':>6}  " + "  ".join(f"{c.name:>10}" for c in clients) + f"  {'all_ok':>7}"
        print(header)
        print("-" * len(header))

    sigma = lo
    while sigma <= hi + 1e-9:
        results = {}
        all_ok  = True
        for spec in clients:
            eps = simulate_epsilon(spec, sigma, num_rounds, target_delta)
            results[spec.name] = eps
            if eps > target_epsilon:
                all_ok = False

        if verbose:
            row = f"{sigma:>6.2f}  "
            row += "  ".join(f"{results[c.name]:>9.4f}ε" for c in clients)
            row += f"  {'✓' if all_ok else '✗':>7}"
            print(row)

        if all_ok and sigma < best_sigma:
            best_sigma   = sigma
            best_results = results
            break                  # first feasible σ is the minimum

        sigma = round(sigma + sigma_step, 10)

    if best_sigma == float("inf"):
        print(f"\n⚠  No σ in [{lo}, {hi}] satisfies ε ≤ {target_epsilon} for all clients.")
        print("   Increase sigma_range or relax target_epsilon.")
    else:
        print(f"\n✓  Recommended noise_multiplier = {best_sigma:.2f}")
        print(f"   Privacy budgets after {num_rounds} rounds (δ={target_delta:.0e}):")
        for name, eps in best_results.items():
            print(f"     {name:>10}:  ε = {eps:.4f}")

    return best_sigma


# ------------------------------------------------------------------
# Your setup — edit these values
# ------------------------------------------------------------------
CLIENTS = [
    ClientSpec(name="client_1", dataset_size=770_415,   batch_size=32, local_epochs=2),
    ClientSpec(name="client_2", dataset_size=1_394_658, batch_size=32, local_epochs=2),
    ClientSpec(name="client_3", dataset_size=10_960,    batch_size=32, local_epochs=2),
]

if __name__ == "__main__":
    recommended_sigma = calibrate(
        clients=CLIENTS,
        target_epsilon=10.0,
        target_delta=1e-5,
        num_rounds=100,
        sigma_range=(0.5, 5.0),
        sigma_step=0.05,
        verbose=True,
    )
    sys.exit(0 if recommended_sigma < float("inf") else 1)
