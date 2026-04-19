import os
import json
import matplotlib.pyplot as plt

METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "poc_workspace/fl_nids/prod_00"))
PLOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../metrics_plots"))
os.makedirs(PLOT_DIR, exist_ok=True)

CLIENTS = ["client0", "client1", "client2"]


def plot_metrics_for_client(client_id):
    metrics_path = os.path.join(METRICS_DIR, client_id, "checkpoints", f"metrics_history_{client_id}.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

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
    out_path = os.path.join(os.getcwd(), "metrics_plots", f"metrics_{client_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metrics plot for {client_id} to {out_path}")


def main():
    for client in CLIENTS:
        try:
            plot_metrics_for_client(client)
        except Exception as e:
            print(f"Failed for {client}: {e}")

if __name__ == "__main__":
    main()
