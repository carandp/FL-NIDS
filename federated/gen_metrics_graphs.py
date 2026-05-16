import os
import json
import matplotlib.pyplot as plt

METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "poc_workspace/fl_nids/prod_00"))
PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "metrics_plots"))
os.makedirs(PLOT_DIR, exist_ok=True)

CLIENTS = ["client0", "client1", "client2"]


def _load_metrics(client_id):
    metrics_path = os.path.join(METRICS_DIR, client_id, "checkpoints", f"metrics_history_{client_id}.json")
    with open(metrics_path, "r") as f:
        return json.load(f)


def plot_metrics_for_client(client_id):
    metrics = _load_metrics(client_id)

    rounds = list(range(len(metrics)))
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["val_loss"] for m in metrics]
    pr_auc = [m["val_pr_auc"] for m in metrics]
    macro_f1 = [m["val_macro_f1"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
    fig.suptitle(f"Metrics for {client_id}", fontsize=16)

    # Loss plot
    axes[0].plot(rounds, train_loss, label="Train Loss", marker='o')
    axes[0].plot(rounds, val_loss, label="Validation Loss", marker='o')
    axes[0].set_title("Loss over Rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # PR_AUC plot
    axes[1].plot(rounds, pr_auc, label="PR_AUC", color="tab:orange", marker='o')
    axes[1].set_title("PR_AUC over Rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("PR_AUC")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Macro F1 plot
    axes[2].plot(rounds, macro_f1, label="Macro F1", color="tab:green", marker='o')
    axes[2].set_title("Macro F1 over Rounds")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(PLOT_DIR, f"metrics_{client_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot for {client_id} to {out_path}")


def plot_metrics_all_clients(client_ids):
    client_metrics = {}
    for client_id in client_ids:
        try:
            client_metrics[client_id] = _load_metrics(client_id)
        except Exception as exc:
            print(f"Failed to load metrics for {client_id}: {exc}")

    if not client_metrics:
        print("No client metrics available for combined plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=300)
    fig.suptitle("Metrics Across Clients", fontsize=16, y=0.93)

    base_colors = plt.cm.tab10.colors
    client_order = list(client_metrics.keys())
    for idx, client_id in enumerate(client_order):
        metrics = client_metrics[client_id]
        rounds = list(range(len(metrics)))
        train_loss = [m["train_loss"] for m in metrics]
        val_loss = [m["val_loss"] for m in metrics]
        pr_auc = [m["val_pr_auc"] for m in metrics]
        macro_f1 = [m["val_macro_f1"] for m in metrics]

        base_color = base_colors[idx % len(base_colors)]
        train_color = (*base_color[:3], 0.35)
        val_color = (*base_color[:3], 0.85)

        axes[0].plot(rounds, train_loss, label=f"{client_id} Train", color=train_color, marker='o')
        axes[0].plot(rounds, val_loss, label=f"{client_id} Val", color=val_color, marker='o')

        axes[1].plot(rounds, pr_auc, label=client_id, color=val_color, marker='o')
        axes[2].plot(rounds, macro_f1, label=client_id, color=val_color, marker='o')

    axes[0].set_title("Loss over Rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend(ncol=2, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("PR_AUC over Rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("PR_AUC")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].set_title("Macro F1 over Rounds")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Macro F1")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(PLOT_DIR, "metrics_all_clients.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined metrics plot to {out_path}")


def main():
    for client in CLIENTS:
        try:
            plot_metrics_for_client(client)
        except Exception as e:
            print(f"Failed for {client}: {e}")

    plot_metrics_all_clients(CLIENTS)

if __name__ == "__main__":
    main()
