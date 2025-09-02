import os
import random
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms

# cartella con dataset originale (adatta al tuo caso)
DATASET_DIR = "E:/MACHINE LEARNING/AIvsREAL_kaggle_dataset/train"
# cartella dove salvare il subset ridotto
OUTPUT_DIR = "E:/MACHINE LEARNING/AIvsREAL_subset/train"

# percentuale di immagini da mantenere
SUBSET_RATIO = 0.1  # 10%

# dimensione finale delle immagini
IMG_SIZE = 512

# trasformazione: ridimensiona + converte in RGB
transform = transforms.Compose([
    #transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


def resize_preserve_aspect(img, target_size=IMG_SIZE):
    w, h = img.size
    if w > h:
        # se l’immagine è più larga → scala la larghezza a target_size
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        # se l’immagine è più alta → scala l’altezza a target_size
        new_h = target_size
        new_w = int(w * target_size / h)
    return img.resize((new_w, new_h), Image.BILINEAR)

def make_subset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # per ogni classe (ad es. real / ai)
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        # cartella di output per questa classe
        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        # lista immagini
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # selezione random di subset
        subset_size = int(len(images) * SUBSET_RATIO)
        subset = random.sample(images, subset_size)  #shuffling delle immagini nel subset

        print(f"Classe {class_name}: {len(images)} → {subset_size} immagini")

        # processa e salva immagini
        for img_name in tqdm(subset):  #loading bar (tqdm)
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_resized=resize_preserve_aspect(img, IMG_SIZE)
                img_t = transforms.ToPILImage()(transform(img_resized))
                img_t.save(os.path.join(output_class_path, img_name))
            except Exception as e:
                print(f"Errore con {img_path}: {e}")

if __name__ == "__main__":
    make_subset()
