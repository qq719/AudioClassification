import os
import math
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import Model
from sklearn.model_selection import KFold

# CONFIG
DATA_DIR = "../WavPreprocessing/png_data"
MODEL_PATH = "./drone_classifier.pth"
BATCH_SIZE = 16
K_FOLDS = 5
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42
IMG_SIZE = (500, 1000)
NUM_CLASSES = 2         # drone / no drone

torch.manual_seed(SEED)
random.seed(SEED)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Initialize model
    model = Model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Setup dataset and kfolds
    kf = KFold(n_splits = K_FOLDS, shuffle = True)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_size = math.floor(len(dataset) * 0.9)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(datasets.ImageFolder(DATA_DIR, transform=transform), [train_size, test_size])

    best_acc = 0.0
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f"[*] Fold: {fold+1}")
        print("----------------------")

        # Get the train and validation sets
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)
        train_loader = DataLoader(dataset = train_subset,
                                  batch_size = BATCH_SIZE,
                                  shuffle = True,
                                  num_workers=4)
        val_loader = DataLoader(dataset = val_subset,
                                batch_size = BATCH_SIZE,
                                shuffle = False,
                                num_workers=4)

        # Start with a fresh model each fold
        model = Model(num_classes = NUM_CLASSES).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss:  {val_loss:.4f} | Val Acc:  {val_acc*100:.2f}%")

            # Save best model
            if val_acc > best_acc:
                torch.save(model.state_dict(), MODEL_PATH)
                best_acc = val_acc
                print(f"[+] Saved new best model ({best_acc*100:.2f}%) to {MODEL_PATH}")

    print(f"[*] Training complete. Best test accuracy: {best_acc*100:.2f}%")
    
if __name__ == "__main__":
    main()
