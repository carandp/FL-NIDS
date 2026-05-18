"""
noise_calculator.py
===================
Offline tool: report each client's privacy budget for a fixed list of
noise multipliers after a fixed number of rounds.
"""
from __future__ import annotations

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


def print_budgets(
    clients: List[ClientSpec],
    noises: List[float],
    target_delta: float,
    num_rounds: int,
    clip_norm: float,
) -> None:
    header = f"{'sigma':>6}  " + "  ".join(f"{c.name:>10}" for c in clients)
    print(header)
    print("-" * len(header))

    for sigma in noises:
        results = {}
        for spec in clients:
            eps = simulate_epsilon(spec, sigma, clip_norm, num_rounds, target_delta)
            results[spec.name] = eps

        row = f"{sigma:>6.2f}  "
        row += "  ".join(f"{results[c.name]:>9.4f}ε" for c in clients)
        print(row)


# ------------------------------------------------------------------
# Your setup — edit these values
# ------------------------------------------------------------------
CLIENTS = [
    ClientSpec(name="client_1", dataset_size=770_415, batch_size=32, local_epochs=2),
    ClientSpec(name="client_2", dataset_size=1_394_658, batch_size=32, local_epochs=2),
    ClientSpec(name="client_3", dataset_size=10_960, batch_size=32, local_epochs=2),
]

NOISES = [0.20, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.05]

TARGET_DELTA = 1e-5
NUM_ROUNDS = 100
CLIP_NORM = 2.0


if __name__ == "__main__":
    print_budgets(
        clients=CLIENTS,
        noises=NOISES,
        target_delta=TARGET_DELTA,
        num_rounds=NUM_ROUNDS,
        clip_norm=CLIP_NORM,
    )
    sys.exit(0)
