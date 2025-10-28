import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import MODEL_PATH, BATCH_SIZE, IMG_SIZE, DATA_DIR, NUM_CLASSES, evaluate, get_data_loaders
from model import Model
from tqdm import tqdm

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
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")
    
if (__name__ == "__main__"):
    main()
