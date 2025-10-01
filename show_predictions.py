import torch
import math
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def show_predictions(test_dataset, test_loader, model, device, num_images=30, row_size=10):
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

    # scegli indici casuali
    indices = random.sample(range(len(test_dataset)), num_images)

    #images, labels = next(iter(test_loader))
    #images, labels = images.to(device), labels.to(device)
    #outputs = model(images)
    #_, preds = torch.max(outputs, 1)

    # Determina quante immagini mostrare
    #num_images = min(num_images, images.size(0))

    num_rows = math.ceil(num_images / row_size)
    plt.figure(figsize=(3 * row_size, 3 * num_rows))

    for i,idx in enumerate(indices):
        # Percorso originale immagine
        img_path, label = test_dataset.samples[idx]
        file_name = os.path.basename(img_path)  # <-- nome file
        img_orig = Image.open(img_path).convert("RGB")
        img_orig = img_orig.resize((512, 512))  # opzionale

        # prepara immagine per la rete (usa la trasformazione del dataset)
        img_tensor = test_dataset[idx][0].unsqueeze(0).to(device)  # (1, 9, H, W)

        # predizione
        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
            pred = pred.item()

        plt.subplot(num_rows, row_size, i + 1)
        plt.imshow(img_orig)
        plt.title(f"{file_name}\nT: {test_dataset.classes[label]} | P: {test_dataset.classes[pred]}",
                  fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

