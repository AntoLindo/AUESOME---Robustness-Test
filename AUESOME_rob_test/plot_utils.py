import matplotlib.pyplot as plt

def plot_training_curves(train_losses, train_accs):
    """
    Mostra le curve di training (loss + accuracy).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Loss
    ax1.plot(train_losses, label="Train Loss", color="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Accuracy
    ax2 = ax1.twinx()
    ax2.plot(train_accs, label="Train Accuracy", color="tab:blue")
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Training Loss & Accuracy")
    fig.tight_layout()
    plt.show()


def plot_validation_curves(val_losses, val_accs):
    """
    Mostra le curve di validation (loss + accuracy).
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Loss
    ax1.plot(val_losses, label="Val Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:orange")
    ax1.tick_params(axis="y", labelcolor="tab:orange")

    # Accuracy
    ax2 = ax1.twinx()
    ax2.plot(val_accs, label="Val Accuracy", color="tab:green")
    ax2.set_ylabel("Accuracy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("Validation Loss & Accuracy")
    fig.tight_layout()
    plt.show()
