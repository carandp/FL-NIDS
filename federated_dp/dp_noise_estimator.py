"""
dp_noise_estimator.py
=====================
Offline tool: given a noise multiplier, report each client's privacy budget
and the maximum epsilon across clients after a fixed number of rounds.

Usage
-----
    uv run dp_noise_estimator --noise 0.20
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List

from jobs.nids_fedavg.app.custom.rdp_accountant import RDPAccountant


@dataclass
class ClientSpec:
    name: str
    dataset_size: int
    batch_size: int = 32
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
    clip_norm: float,
    num_rounds: int,
    delta: float,
) -> float:
    acc = RDPAccountant()
    for _ in range(num_rounds):
        acc.step(
            noise_multiplier=noise_multiplier,
            sample_rate=spec.sample_rate,
            clip_norm=clip_norm,
            num_steps=spec.steps_per_round,
        )
    return acc.get_epsilon(delta)


def estimate_max_epsilon(
    clients: List[ClientSpec],
    noise_multiplier: float,
    target_delta: float = 1e-5,
    num_rounds: int = 100,
    clip_norm: float = 2.0,
) -> float:
    results = {}
    for spec in clients:
        eps = simulate_epsilon(spec, noise_multiplier, clip_norm, num_rounds, target_delta)
        results[spec.name] = eps

    header = f"{'σ':>6}  " + "  ".join(f"{c.name:>10}" for c in clients) + f"  {'max':>10}"
    print(header)
    print("-" * len(header))
    row = f"{noise_multiplier:>6.2f}  "
    row += "  ".join(f"{results[c.name]:>9.4f}ε" for c in clients)
    row += f"  {max(results.values()):>9.4f}ε"
    print(row)

    print(
        f"\nMax ε after {num_rounds} rounds (δ={target_delta:.0e}): {max(results.values()):.4f}"
    )

    return max(results.values())


# ------------------------------------------------------------------
# Your setup — edit these values
# ------------------------------------------------------------------
CLIENTS = [
    ClientSpec(name="client_0", dataset_size=770_415, batch_size=32, local_epochs=2),
    ClientSpec(name="client_1", dataset_size=1_394_658, batch_size=32, local_epochs=2),
    ClientSpec(name="client_2", dataset_size=10_960, batch_size=32, local_epochs=2),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate the maximum epsilon across clients for a given noise multiplier.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        required=True,
        help="Noise multiplier (sigma).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Target delta for the privacy budget (default: 1e-5).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of FL rounds (default: 100).",
    )
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=2.0,
        help="Clip norm (default: 2.0).",
    )
    args = parser.parse_args()

    estimate_max_epsilon(
        clients=CLIENTS,
        noise_multiplier=args.noise,
        target_delta=args.delta,
        num_rounds=args.rounds,
        clip_norm=args.clip_norm,
    )
    sys.exit(0)
