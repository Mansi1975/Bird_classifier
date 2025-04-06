import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# 1. Data Preparation
# ----------------------------

import re

def extract_number(f):
    match = re.search(r'\d+', f)
    return int(match.group()) if match else float('inf')

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted( os.listdir(root_dir), key=extract_number)
            #[f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))],
            #key=lambda x: int(os.path.splitext(x)[0])
        
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading {img_name}, skipping...")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# ----------------------------
# 2. Data Transforms
# ----------------------------

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------
# 3. Model Setup
# ----------------------------

def create_model(num_classes=200):
    #model = models.efficientnet_b4(pretrained=True)
    from torchvision.models import EfficientNet_B4_Weights
    weights = EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
    )
    return model
    
# ----------------------------
# 4. Training Loop
# ----------------------------

def train_model():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    
    # Data Loaders
    train_dataset = ImageFolder(root='../data/train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Test Loader
    test_dataset = TestDataset(root_dir='../data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training Loop
    for epoch in range(1):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")
    
    return model

# ----------------------------
# 5. Submission Generation
# ----------------------------

def generate_submission(model, test_loader, device):
    model.eval()
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            filenames.extend(names)
    
    # Create mapping
    class_to_idx = ImageFolder(root='../data/train').class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': filenames,
        'label': [idx_to_class[p] for p in predictions]
    })
    
    # Sort numerically
    df['num'] = df['ID'].apply(lambda x: int(os.path.splitext(x)[0]))
    df = df.sort_values('num').drop('num', axis=1)
    df.to_csv('../submission.csv', index=False)

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    trained_model = train_model()
    test_loader = DataLoader(
        TestDataset(root_dir='../data/test', transform=test_transform),
        batch_size=32, shuffle=False, num_workers=4
    )
    generate_submission(trained_model, test_loader, 
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))