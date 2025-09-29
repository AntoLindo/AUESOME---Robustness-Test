import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """
    Mostra i grafici dell'andamento di loss e accuracy durante il training.

    Args:
        train_losses (list): valori di training loss per epoca
        val_losses (list): valori di validation loss per epoca
        train_accs (list, opzionale): valori di training accuracy per epoca
        val_accs (list, opzionale): valori di validation accuracy per epoca
    """
    plt.figure(figsize=(10, 4))

    # 🔹 Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Trend")
    plt.legend()

    # 🔹 Accuracy (se fornita)
    if train_accs is not None and val_accs is not None:
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Trend")
        plt.legend()

    plt.tight_layout()
    plt.show()
