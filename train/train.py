import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
import warnings

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SportPictureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SportImageClassifierEfficientNet(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(SportImageClassifierEfficientNet, self).__init__()

        # Load pre-trained Efficient Net
        self.base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Remove original classifier
        self.base_model.classifier = nn.Identity()

        # Add custom layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x


def get_transforms():
    """Get train, validation, and test transforms"""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, val_transforms


def train_model(
    train_dataset,
    val_dataset,
    hyperparam_dict,
    num_epochs=30,
    num_classes=100,
    patience=5,
    save_path='best_model.pth',
    onnx_path='best_sport-image-classifier-model.onnx'
):
    """Train the model with given hyperparameters"""
    
    print(f"Training with hyperparameters: {hyperparam_dict}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparam_dict['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparam_dict['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    model = SportImageClassifierEfficientNet(
        num_classes=num_classes,
        dropout_rate=hyperparam_dict['dropout_rate']
    )
    model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hyperparam_dict['lr'],
        weight_decay=hyperparam_dict['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2
    )
    
    # Training metrics
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ')
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        scheduler.step(val_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparameters': hyperparam_dict,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f'  ✓ Saved best model (Val Acc: {val_acc:.4f})')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            print(f'Best validation accuracy: {best_val_acc:.4f}')
            break
        
        torch.cuda.empty_cache()
    
    # Load best model and export to ONNX
    print(f'\nLoading best model from {save_path}')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export to ONNX
    print(f'Exporting model to ONNX format: {onnx_path}')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        print(f'✓ ONNX model saved: {onnx_path}')
    except Exception as e:
        print(f'⚠ Failed to save ONNX: {e}')
    
    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


def main():
    """Main training script"""
    
    # Configuration
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    
    # Hyperparameters
    hyperparam_dict = {
        'weight_decay': 0.01,
        'lr': 0.001,
        'dropout_rate': 0.2,
        'batch_size': 128
    }
    
    num_epochs = 30
    num_classes = 100
    
    # Get transforms
    train_transforms, val_transforms = get_transforms()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SportPictureDataset(
        data_dir=data_dir / "train",
        transform=train_transforms
    )
    
    val_dataset = SportPictureDataset(
        data_dir=data_dir / "valid",
        transform=val_transforms
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Train model
    results = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hyperparam_dict=hyperparam_dict,
        num_epochs=num_epochs,
        num_classes=num_classes,
        patience=5,
        save_path='best_model.pth',
        onnx_path='best_sport-image-classifier-model.onnx'
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.4f} ({results['best_val_acc']*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
