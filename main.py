import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import MODEL_PATH, BATCH_SIZE, IMG_SIZE, DATA_DIR, NUM_CLASSES, get_data_loaders
from model import Model
from tqdm import tqdm

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    precision = tp / (tp + fn) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return epoch_loss, epoch_acc, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 4)
    
    model = Model(NUM_CLASSES)
    model.to(device)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    loss, acc, precision, recall, f1 = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    print(f"[*] Loss = {loss:.4f}")
    print(f"[*] Accuracy = {acc*100:.2f}%")
    print(f"[*] Precision = {precision:.4f}")
    print(f"[*] Recall = {recall:.4f}")
    print(f"[*] F1 = {f1:.4f}")
    
if (__name__ == "__main__"):
    main()
