import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. Trasformazioni immagini ---
IMG_SIZE = 512  # se vuoi lavorare su subset ridotto
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # mean ImageNet
                         [0.229, 0.224, 0.225])  # std ImageNet
])

# --- 2. Caricamento dataset ---
train_dataset = datasets.ImageFolder("E:/MACHINE LEARNING/AIvsREAL_subset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder("E:/MACHINE LEARNING/AIvsREAL_subset/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- 3. Caricamento ResNet18 pre-addestrata ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

# --- 4. Adattare l'ultimo layer al tuo problema ---
num_classes = 2  # AI vs Real
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# --- 5. Loss e optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- 6. Esempio training loop minimale ---
for epoch in range(2):  # per test iniziale, poi aumenti
    model.train()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

summary(model,32,3,512,512)
