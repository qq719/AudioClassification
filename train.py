import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from main import Model, get_device

# CONFIG
DATA_DIR = "./png_data"
MODEL_PATH = "./drone_classifier.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42
IMG_SIZE = (500, 1000)
NUM_CLASSES = 2         # drone / no drone

torch.manual_seed(SEED)
random.seed(SEED)

def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    total_size = len(dataset)
    test_size = total_size // 2
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"[*] Training samples: {len(train_dataset)} | Testing samples: {len(test_dataset)}")
    return train_loader, test_loader

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    device = get_device()
    print(f"[*] Using device: {device}")

    # Initialize model
    model = Model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_data_loaders()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")

        # Save best model
        if test_acc > best_acc:
            torch.save(model.state_dict(), MODEL_PATH)
            best_acc = test_acc
            print(f"[+] Saved new best model ({best_acc*100:.2f}%) to {MODEL_PATH}")

    print(f"[*] Training complete. Best test accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
