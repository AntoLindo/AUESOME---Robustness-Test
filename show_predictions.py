import torch
import matplotlib.pyplot as plt
from PIL import Image

def show_predictions(test_dataset, test_loader, model, device, num_images=8):
    """
    Visualizza un batch di immagini originali con le predizioni del modello.

    Args:
        test_dataset: dataset di test (ImageFolder)
        test_loader: DataLoader associato
        model: modello 9-canali già caricato e in eval()
        device: "cuda" o "cpu"
        num_images: numero di immagini da visualizzare
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    # Determina quante immagini mostrare
    num_images = min(num_images, images.size(0))

    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        # Percorso originale immagine
        img_path, _ = test_dataset.samples[i]
        img_orig = Image.open(img_path).convert("RGB")
        img_orig = img_orig.resize((224, 224))  # opzionale

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_orig)
        plt.title(f"True: {test_dataset.classes[labels[i]]}\nPred: {test_dataset.classes[preds[i]]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
