# %%
# ==================================================
# Import
# ==================================================

# Utility
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from timeit import default_timer as timer
from tqdm.auto import tqdm
from PIL import Image
import warnings
warnings.filterwarnings(action = 'ignore')

# Pytorch
import torch
from torch import nn
from torch.amp import autocast, GradScaler
import torchvision
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from scipy.optimize import minimize

# Versions of Torch and Torchvision
if __name__ == '__main__':
    print(f'Torch version: {torch.__version__}')
    print(f'Torchvision version: {torchvision.__version__}')

# Device Configuration
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])

# Seed Configuration
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %%
# ==================================================
# Exploration
# ==================================================

# Folder Exploration
def check_path(target_path):
    for root, folders, files in os.walk(target_path):
        print(f'There are {len(folders)} folders and {len(files)} images in \'{root}\'.')

if __name__ == '__main__':
    train_path = 'E:/PYTHON_FILES/dacon_gravel/train'
    test_path = 'E:/PYTHON_FILES/dacon_gravel/test'

    check_path(train_path)
    check_path(test_path)

# %%
# ==================================================
# Configuration
# ==================================================

# ConvNeXt Base Configuration
if __name__ == '__main__':
    CFG = {
        'SEED': 42,
        'IMG_SIZE': 224,
        'VALIDATION_SIZE': 0.25,
        'BATCH_SIZE': 32,
        'EPOCH': 30,
        'LEARNING_RATE': 1e-4,
    }

# Swin V2 Small Configuration
if __name__ == '__main__':
    CFG2 = {
        'SEED': 42,
        'IMG_SIZE': 256,
        'VALIDATION_SIZE': 0.25,
        'BATCH_SIZE': 32,
        'EPOCH': 40,
        'LEARNING_RATE': 5e-5,
    }

# EfficientNet V2 Medium Configuration
if __name__ == '__main__':
    CFG3 = {
        'SEED': 42,
        'IMG_SIZE': 224,
        'VALIDATION_SIZE': 0.25,
        'BATCH_SIZE': 32,
        'EPOCH': 40,
        'LEARNING_RATE': 1e-4,
    }

# Seed Configuration
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(CFG['SEED'])

# %%
# ==================================================
# Preprocessing
# ==================================================

# Train-Test Split
if __name__ == '__main__':
    full_path_list = list(Path(train_path).glob('*/*.jpg'))
    full_df = pd.DataFrame(columns = ['path', 'class'])
    full_df['path'] = full_path_list
    full_df['path'] = full_df['path'].astype(str)
    full_df['class'] = full_df['path'].apply(lambda x: str(x).split('\\')[4])

    print('\nData Before Split:\n', full_df.head())

    train_df, val_df = train_test_split(full_df,
                                        test_size = CFG['VALIDATION_SIZE'],
                                        shuffle = True,
                                        random_state= CFG['SEED'],
                                        stratify = full_df['class'])

    class_encoder = LabelEncoder()
    train_df['class_index'] = class_encoder.fit_transform(train_df['class'])
    val_df['class_index'] = class_encoder.transform(val_df['class'])

    print('\nTrain Data:\n', train_df.head())
    print('\nValidation Data:\n', val_df.head())

    num_class = full_df['class'].nunique()

# Pad-to-Square Transformation
class PadSquare:
    def __init__(self, fill = 0):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        max_side = max(width, height)

        pad_left = (max_side - width) // 2
        pad_right = (max_side - width) - pad_left
        pad_top = (max_side - height) // 2
        pad_bottom = (max_side - height) - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return transforms.functional.pad(image, padding, fill = self.fill)

# 224 * 224 Transformation
if __name__ == '__main__':
    train_transform = transforms.Compose([
        PadSquare(),
        transforms.RandomResizedCrop(size = (CFG['IMG_SIZE'], CFG['IMG_SIZE']),
                                     scale = (0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.GaussianBlur(kernel_size = 5, sigma = (0.1, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        PadSquare(),
        transforms.Resize(size = (CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# 256 * 256 Transformation
if __name__ == '__main__':
    train_transform2 = transforms.Compose([
        PadSquare(),
        transforms.RandomResizedCrop(size = (CFG2['IMG_SIZE'], CFG2['IMG_SIZE']),
                                     scale = (0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.GaussianBlur(kernel_size = 5, sigma = (0.1, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transform2 = transforms.Compose([
        PadSquare(),
        transforms.Resize(size = (CFG2['IMG_SIZE'], CFG2['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, path_list, class_list, transform):
        self.path_list = path_list
        self.class_list = class_list
        self.transform = transform
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        image_path = self.path_list[index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.class_list is not None:
            class_ = self.class_list[index]
            return image, class_
        
        else:
            return image

# 224 * 224 Datasets
if __name__ == '__main__':
    train_dataset = CustomDataset(path_list = train_df['path'].values,
                                class_list = train_df['class_index'].values,
                                transform = train_transform)

    val_dataset = CustomDataset(path_list = val_df['path'].values,
                                class_list = val_df['class_index'].values,
                                transform = test_transform)
    
# 256 * 256 Dataset
if __name__ == '__main__':
    train_dataset2 = CustomDataset(path_list = train_df['path'].values,
                                class_list = train_df['class_index'].values,
                                transform = train_transform2)

    val_dataset2 = CustomDataset(path_list = val_df['path'].values,
                                class_list = val_df['class_index'].values,
                                transform = test_transform2)

# 224 * 224 Train & Validation Dataloader
if __name__ == '__main__':
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = CFG['BATCH_SIZE'],
                              num_workers = 6,
                              shuffle = False,
                              pin_memory = True,
                              prefetch_factor = 4
                              )

    val_loader = DataLoader(dataset = val_dataset,
                            batch_size = CFG['BATCH_SIZE'],
                            num_workers = 6,
                            shuffle = False,
                            pin_memory = True,
                            prefetch_factor = 4
                            )

# 256 * 256 Train & Validation Dataloader
if __name__ == '__main__':
    train_loader2 = DataLoader(dataset = train_dataset2,
                              batch_size = CFG2['BATCH_SIZE'],
                              num_workers = 6,
                              shuffle = False,
                              pin_memory = True,
                              prefetch_factor = 4
                              )

    val_loader2 = DataLoader(dataset = val_dataset2,
                            batch_size = CFG2['BATCH_SIZE'],
                            num_workers = 6,
                            shuffle = False,
                            pin_memory = True,
                            prefetch_factor = 4
                            )
    
# 224 * 224 Test Loader
if __name__ == '__main__':
    test_path_list = pd.Series(Path(test_path).glob('./*.jpg'))
    test_path_list = test_path_list.astype(str).tolist()

    test_dataset = CustomDataset(path_list = test_path_list,
                                 class_list = None,
                                 transform = test_transform)
    
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = CFG['BATCH_SIZE'],
                            num_workers = 6,
                            shuffle = False,
                            pin_memory = True,
                            prefetch_factor = 4
                            )

# 256 * 256 Test Loader
if __name__ == '__main__':
    test_dataset2 = CustomDataset(path_list = test_path_list,
                                 class_list = None,
                                 transform = test_transform2)
    
    test_loader2 = DataLoader(dataset = test_dataset2,
                            batch_size = CFG2['BATCH_SIZE'],
                            num_workers = 6,
                            shuffle = False,
                            pin_memory = True,
                            prefetch_factor = 4
                            )
    
#%%
# ==================================================
# Model Selection Tools
# ==================================================

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.0, alpha = None, reduction = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, true):
        log_prob = nn.functional.log_softmax(pred, dim = 1)
        prob = torch.exp(log_prob)
        true_one_hot = nn.functional.one_hot(true, num_classes = pred.size(1)).float()

        if self.alpha is not None:
            alpha = self.alpha[true].unsqueeze(1)
        else:
            alpha = 1.0

        loss = -alpha * ((1 - prob) ** self.gamma) * log_prob
        loss = (loss * true_one_hot).sum(dim = 1)

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

# Weights for Focal Loss
if __name__ == '__main__':
    class_weight_array = compute_class_weight(class_weight = 'balanced',
                                              classes = np.unique(train_df['class_index']),
                                              y = train_df['class_index'])
    class_weight_tensor = torch.tensor(class_weight_array, dtype=torch.float).to(device)
    
# Validation Function
def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss_list = []
    pred_list, true_list = [], []

    with torch.no_grad():
        for x, y in tqdm(iter(val_loader), desc = 'Validation'):
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            
            loss = criterion(pred, y)
            
            pred_list += pred.argmax(1).detach().cpu().numpy().tolist()
            true_list += y.detach().cpu().numpy().tolist()
            
            val_loss_list.append(loss.item())
        
        val_loss = np.mean(val_loss_list)
        val_score = f1_score(true_list, pred_list, average = 'macro')
    
    return val_loss, val_score

def train(model, EPOCH, criterion, optimizer, scheduler, scaler, train_loader, val_loader, device, best_path, checkpoint_path = None):
    
    model.to(device)
    criterion = criterion.to(device)

    best_score = 0
    early_stop_counter = 0
    patience = 5
    start_epoch = 1

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,
                                weights_only = False,
                                map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        early_stop_counter = checkpoint['early_stop_counter']

        print(f"Resumed from epoch {checkpoint['epoch']}.")

    start_time = timer()

    for epoch in range(start_epoch, EPOCH + 1):
        
        model.train()
        train_loss_list = []

        for x, y in tqdm(iter(train_loader), desc=f'Epoch {epoch}'):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with autocast(device_type = 'cuda'):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_list.append(loss.item())

        val_loss, val_score = validation(model, val_loader, criterion, device)
        train_loss = np.mean(train_loss_list)
        
        print(f'Train Loss: {train_loss:.5f} \nValidation Loss: {val_loss:.5f} \nValidation Macro F1: {val_score:.5f}')

        if scheduler is not None:
            scheduler.step(val_score)
            print('Learning Rate : {:.10f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        if val_score > best_score:
            best_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved (epoch {epoch}, F1 score {val_score:.5f}): {best_path}")
        else:
            early_stop_counter += 1
            print(f"Early Stop Counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        if checkpoint_path is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_score': best_score,
                'early_stop_counter': early_stop_counter
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        end_time = timer()
        print(f'\nTime elapsed: {(end_time - start_time)/60:.2f} minutes')
        print('--------------------------------------------------')

    return 

# %%
# ==================================================
# Model Selection: ConvNeXt Base (1)
# ==================================================

# Backbone
if __name__ == '__main__':
    model = models.convnext_base(weights = 'ConvNeXt_Base_Weights.DEFAULT')
    model.classifier[2] = nn.Linear(in_features = model.classifier[2].in_features,
                         out_features = num_class)
    
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    
# Criterion
if __name__ == '__main__':
    criterion = FocalLoss(gamma = 2.0, alpha = class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer = torch.optim.AdamW(model.parameters(), lr = CFG['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler = GradScaler()

# Model Selection (1)
if __name__ == '__main__':
    best_path = 'E:/PYTHON_FILES/dacon_gravel/best_convnext.pt'
    checkpoint_path = 'E:/PYTHON_FILES/dacon_gravel/convnext_chkpt.pth'
    
    seed_everything(CFG['SEED'])
    model_trained = train(model = model,
                          EPOCH = CFG['EPOCH'] - 15,
                          criterion = criterion,
                          optimizer = optimizer,
                          scheduler = scheduler,
                          scaler = scaler,
                          train_loader = train_loader,
                          val_loader = val_loader,
                          device = device,
                          best_path = best_path,
                          checkpoint_path = checkpoint_path
                          )

# %%
# ==================================================
# Model Selection: ConvNeXt Base (2)
# ==================================================

# Backbone: ConvNeXt Base
if __name__ == '__main__':
    model = models.convnext_base(weights = None)
    model.classifier[2] = nn.Linear(in_features = model.classifier[2].in_features,
                         out_features = num_class)
    
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    
# Criterion
if __name__ == '__main__':
    criterion = FocalLoss(gamma = 2.0, alpha = class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer = torch.optim.AdamW(model.parameters(), lr = CFG['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler = GradScaler()

# Model Selection (2)
if __name__ == '__main__':
    best_path = 'E:/PYTHON_FILES/dacon_gravel/best_convnext.pt'
    checkpoint_path = 'E:/PYTHON_FILES/dacon_gravel/convnext_chkpt.pth'

    seed_everything(CFG['SEED'])
    model_trained = train(model = model,
                          EPOCH = CFG['EPOCH'],
                          criterion = criterion,
                          optimizer = optimizer,
                          scheduler = scheduler,
                          scaler = scaler,
                          train_loader = train_loader,
                          val_loader = val_loader,
                          device = device,
                          best_path = best_path,
                          checkpoint_path = checkpoint_path
                          )
    
# %%
# ==================================================
# Model Selection: Swin Small (1)
# ==================================================

# Backbone: Swin small
if __name__ == '__main__':
    model2 = models.swin_v2_s(weights = 'Swin_V2_S_Weights.DEFAULT')
    model2.head = nn.Linear(in_features = model2.head.in_features,
                         out_features = num_class)
    
    for param in model2.parameters():
        param.requires_grad = True

    model2.to(device)

# Criterion
if __name__ == '__main__':
    criterion2 = FocalLoss(gamma = 2.0, alpha = class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr = CFG2['LEARNING_RATE'])
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer2,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler2 = GradScaler()

# Model Selection
if __name__ == '__main__':
    best_path2 = 'E:/PYTHON_FILES/dacon_gravel/best_swin.pt'
    checkpoint_path2 = 'E:/PYTHON_FILES/dacon_gravel/swin_chkpt.pth'

    seed_everything(CFG2['SEED'])
    model_trained2 = train(model = model2,
                          EPOCH = CFG2['EPOCH'] - 30,
                          criterion = criterion2,
                          optimizer = optimizer2,
                          scheduler = scheduler2,
                          scaler = scaler2,
                          train_loader = train_loader2,
                          val_loader = val_loader2,
                          device = device,
                          best_path = best_path2,
                          checkpoint_path = checkpoint_path2
                          )

# %%
# ==================================================
# Model Selection: Swin Small (2)
# ==================================================

# Backbone: Swin Small
if __name__ == '__main__':
    model2 = models.swin_v2_s(weights = None)
    model2.head = nn.Linear(in_features = model2.head.in_features,
                         out_features = num_class)
    
    for param in model2.parameters():
        param.requires_grad = True
        
    model2.to(device)

# Criterion
if __name__ == '__main__':
    criterion2 = FocalLoss(gamma = 2.0, alpha = class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr = CFG2['LEARNING_RATE'])
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer2,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler2 = GradScaler()

# Model Selection
if __name__ == '__main__':
    best_path2 = 'E:/PYTHON_FILES/dacon_gravel/best_swin.pt'
    checkpoint_path2 = 'E:/PYTHON_FILES/dacon_gravel/swin_chkpt.pth'

    seed_everything(CFG2['SEED'])
    model_trained2 = train(model = model2,
                          EPOCH = CFG2['EPOCH'] - 20,
                          criterion = criterion2,
                          optimizer = optimizer2,
                          scheduler = scheduler2,
                          scaler = scaler2,
                          train_loader = train_loader2,
                          val_loader = val_loader2,
                          device = device,
                          best_path = best_path2,
                          checkpoint_path = checkpoint_path2
                          )

# %%
# ==================================================
# Model Selection: Swin Small (3)
# ==================================================

# Backbone: Swin Small
if __name__ == '__main__':
    model2 = models.swin_v2_s(weights = None)
    model2.head = nn.Linear(in_features = model2.head.in_features,
                         out_features = num_class)
    
    for param in model2.parameters():
        param.requires_grad = True
        
    model2.to(device)

# Criterion
if __name__ == '__main__':
    criterion2 = FocalLoss(gamma=2.0, alpha=class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr = CFG2['LEARNING_RATE'])
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer2,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler2 = GradScaler()

# Model Selection
if __name__ == '__main__':
    best_path2 = 'E:/PYTHON_FILES/dacon_gravel/best_swin.pt'
    checkpoint_path2 = 'E:/PYTHON_FILES/dacon_gravel/swin_chkpt.pth'

    seed_everything(CFG2['SEED'])
    model_trained2 = train(model = model2,
                          EPOCH = CFG2['EPOCH'] - 10,
                          criterion = criterion2,
                          optimizer = optimizer2,
                          scheduler = scheduler2,
                          scaler = scaler2,
                          train_loader = train_loader2,
                          val_loader = val_loader2,
                          device = device,
                          best_path = best_path2,
                          checkpoint_path = checkpoint_path2
                          )
    
# %%
# ==================================================
# Model Selection: Swin Small (4)
# ==================================================

# Backbone: Swin Small
if __name__ == '__main__':
    model2 = models.swin_v2_s(weights = None)
    model2.head = nn.Linear(in_features = model2.head.in_features,
                         out_features = num_class)
    
    for param in model2.parameters():
        param.requires_grad = True
        
    model2.to(device)

# Criterion
if __name__ == '__main__':
    criterion2 = FocalLoss(gamma=2.0, alpha=class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr = CFG2['LEARNING_RATE'])
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer2,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler2 = GradScaler()

# Model Selection
if __name__ == '__main__':
    best_path2 = 'E:/PYTHON_FILES/dacon_gravel/best_swin.pt'
    checkpoint_path2 = 'E:/PYTHON_FILES/dacon_gravel/swin_chkpt.pth'

    seed_everything(CFG2['SEED'])
    model_trained2 = train(model = model2,
                          EPOCH = CFG2['EPOCH'],
                          criterion = criterion2,
                          optimizer = optimizer2,
                          scheduler = scheduler2,
                          scaler = scaler2,
                          train_loader = train_loader2,
                          val_loader = val_loader2,
                          device = device,
                          best_path = best_path2,
                          checkpoint_path = checkpoint_path2
                          )
    
#%%
# ==================================================
# Model Selection: EfficientNetV2 Medium (1)
# ==================================================

# Backbone: EfficientNetV2 Medium
if __name__ == '__main__':
    model3 = models.efficientnet_v2_m(weights = 'EfficientNet_V2_M_Weights.DEFAULT')
    model3.classifier[1] = nn.Linear(in_features = model3.classifier[1].in_features,
                         out_features = num_class)
    
    for param in model3.parameters():
        param.requires_grad = True

    model3.to(device)

# Criterion
if __name__ == '__main__':
    criterion3 = FocalLoss(gamma = 2.0, alpha = class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr = CFG3['LEARNING_RATE'])
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer3,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler3 = GradScaler()

# Model Selection
if __name__ == '__main__':
    best_path3 = 'E:/PYTHON_FILES/dacon_gravel/best_efficientnet.pt'
    checkpoint_path3 = 'E:/PYTHON_FILES/dacon_gravel/efficientnet_chkpt.pth'

    seed_everything(CFG3['SEED'])
    model_trained3 = train(model = model3,
                          EPOCH = CFG3['EPOCH'] - 20,
                          criterion = criterion3,
                          optimizer = optimizer3,
                          scheduler = scheduler3,
                          scaler = scaler3,
                          train_loader = train_loader,
                          val_loader = val_loader,
                          device = device,
                          best_path = best_path3,
                          checkpoint_path = checkpoint_path3
                          )

# %%
# ==================================================
# Model Selection: EfficientNetV2 Medium (2)
# ==================================================

# Backbone: EfficientNetV2 Medium
if __name__ == '__main__':
    model3 = models.efficientnet_v2_m(weights = None)
    model3.classifier[1] = nn.Linear(in_features = model3.classifier[1].in_features,
                         out_features = num_class)
    
    for param in model3.parameters():
        param.requires_grad = True

    model3.to(device)

# Criterion
if __name__ == '__main__':
    criterion3 = FocalLoss(gamma=2.0, alpha=class_weight_tensor)

# Optimizer & Scheduler
if __name__ == '__main__':
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr = CFG3['LEARNING_RATE'])
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer3,
                                                           mode = 'max',
                                                           factor = 0.5,
                                                           patience = 2,
                                                           min_lr = 1e-8)
    
# Scaler for AMP
if __name__ == '__main__':
    scaler3 = GradScaler()

# Model Selection (2)
if __name__ == '__main__':
    best_path3 = 'E:/PYTHON_FILES/dacon_gravel/best_efficientnet.pt'
    checkpoint_path3 = 'E:/PYTHON_FILES/dacon_gravel/efficientnet_chkpt.pth'

    seed_everything(CFG3['SEED'])
    model_trained3 = train(model = model3,
                          EPOCH = CFG3['EPOCH'],
                          criterion = criterion3,
                          optimizer = optimizer3,
                          scheduler = scheduler3,
                          scaler = scaler3,
                          train_loader = train_loader,
                          val_loader = val_loader,
                          device = device,
                          best_path = best_path3,
                          checkpoint_path = checkpoint_path3
                          )

#%%
#======================================================================
# Ensemble Tools
#======================================================================

# Inference Function
def inference(model, dataloader, device, validation = False):
    model.eval()
    prob_list = []
    pred_list = []
    true_list = []

    if validation == True:
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc='Inference'):
                x, y = x.to(device), y.to(device)

                output = model(x)
                prob = nn.functional.softmax(output, dim = 1)
                pred = prob.argmax(dim = 1)

                prob_list.append(prob.cpu().numpy())            
                pred_list.append(pred.cpu().numpy())
                true_list.append(y.cpu().numpy())

        prob_array = np.concatenate(prob_list, axis = 0)    # (N, C)
        pred = np.concatenate(pred_list, axis = 0)    # (N, C)
        true = np.concatenate(true_list, axis = 0)    # (N,  )

        return prob_array, pred, true
    
    else:
        with torch.no_grad():
            for x in tqdm(dataloader, desc='Inference'):
                x = x.to(device)

                output = model(x)
                prob = nn.functional.softmax(output, dim = 1)
                pred = prob.argmax(dim = 1)

                prob_list.append(prob.cpu().numpy())            
                pred_list.append(pred.cpu().numpy())

        prob_array = np.concatenate(prob_list, axis = 0)    # (N, C)
        pred = np.concatenate(pred_list, axis = 0)    # (N, C)

        return prob_array, pred

# Focal Loss for NumPy
def focal_loss_np(prob_array, true, gamma = 2.0, alpha = None, eps = 1e-8):
    prob_clipped = np.clip(prob_array, eps, 1.0 - eps)
    log_prob = np.log(prob_clipped)
    focal_factor = (1 - prob_clipped) ** gamma

    N, C = prob_array.shape
    true_one_hot = np.eye(C)[true]

    if alpha is not None:
        alpha_factor = alpha[true]
    else:
        alpha_factor = 1.0

    loss = -alpha_factor * np.sum(true_one_hot * focal_factor * log_prob, axis = 1)
    return np.mean(loss)

# Ensemble Focal Loss
def ensemble_focal_loss(weights, true, alpha, *args):
    '''
    weights: (n_model, ) numpy array (sum-to-one)
    args: (prob1_array, prob2_array, ..., probN_array)
    '''
    prob_arrays = args

    if any(w < 0 or w > 1 for w in weights) or abs(np.sum(weights) - 1) > 1e-5:
        return np.inf

    stacked_prob_arrays = np.stack(prob_arrays, axis = 0)
    ensemble_prob = np.tensordot(weights, stacked_prob_arrays, axes = 1)

    return focal_loss_np(ensemble_prob, true, gamma = 2.0, alpha = alpha)

# %%
#======================================================================
# Best Models
#======================================================================

# Validation Score: ConvNeXt Base
import gc

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    model = models.convnext_base(weights = None)
    model.classifier[2] = nn.Linear(in_features = model.classifier[2].in_features,
                         out_features = num_class)
    
    model_state_dict = torch.load('E:/PYTHON_FILES/dacon_gravel/best_convnext.pt',
                                  map_location = device)
    model.load_state_dict(model_state_dict)
    model.to(device)

    prob_array, pred, true = inference(model, val_loader, device, validation = True)
    val_score = f1_score(true, pred, average = 'macro')
    print(f'Validation F1 Score of ConvNeXt Base: {val_score}\n')

# Test: ConvNeXt Base
    gc.collect()
    torch.cuda.empty_cache()
    test_prob, _ = inference(model, test_loader, device)

# Validation Score: Swin V2 Small
if __name__ == '__main__':
    gc.collect()
    del model
    torch.cuda.empty_cache()
    model2 = models.swin_v2_s(weights = None)
    model2.head = nn.Linear(in_features = model2.head.in_features,
                         out_features = num_class)

    model2_state_dict = torch.load('E:/PYTHON_FILES/dacon_gravel/best_swin.pt',
                                  map_location = device)
    model2.load_state_dict(model2_state_dict)
    model2.to(device)

    prob_array2, pred2, _ = inference(model2, val_loader2, device, validation = True)
    val_score2 = f1_score(true, pred2, average = 'macro')
    print(f'Validation F1 Score of Swin V2 Small: {val_score2}\n')

# Test: Swin V2 Small
    gc.collect()
    torch.cuda.empty_cache()
    test_prob2, _ = inference(model2, test_loader2, device)

# Validation Score: EfficientNet V2 Medium 
if __name__ == '__main__':
    gc.collect()
    del model2
    torch.cuda.empty_cache()
    model3 = models.efficientnet_v2_m(weights = None)
    model3.classifier[1] = nn.Linear(in_features = model3.classifier[1].in_features,
                         out_features = num_class)
    
    model3_state_dict = torch.load('E:/PYTHON_FILES/dacon_gravel/best_efficientnet.pt',
                                  map_location = device)
    model3.load_state_dict(model3_state_dict)
    model3.to(device)

    prob_array3, pred3, _ = inference(model3, val_loader, device, validation = True)
    val_score3 = f1_score(true, pred3, average = 'macro')
    print(f'Validation F1 Score of EfficientNet V2 Medium: {val_score3}\n')

# Test: EfficientNet V2 Medium
    gc.collect()
    torch.cuda.empty_cache()
    test_prob3, _ = inference(model3, test_loader, device)

# %%
#======================================================================
# Ensemble
#======================================================================

# Optimize Weights
if __name__ == '__main__':
    initial_w = [1/3, 1/3, 1/3]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1), (0, 1), (0, 1)]

    optimize_w = minimize(ensemble_focal_loss,
                        x0 = initial_w,
                        args = (true, class_weight_array, prob_array, prob_array2, prob_array3),
                        method = 'SLSQP',
                        bounds = bounds,
                        constraints = constraints)

    w = optimize_w.x

    print(f'Optimal Weights: {w}\n')

# Validation Score: Ensemble Model
if __name__ == '__main__':
    ensemble_prob_array = w[0] * prob_array + w[1] * prob_array2 + w[2] * prob_array3
    ensemble_pred = np.argmax(ensemble_prob_array, axis = 1)
    ensemble_val_score = f1_score(true, ensemble_pred, average = 'macro')

    print(f'Validation F1 Score of Ensemble Model: {ensemble_val_score}')
    print('Note that the weights were optimized using the validation set.')

# Test: Ensemble Model
if __name__ == '__main__':
    ensemble_test_prob = w[0] * test_prob + w[1] * test_prob2 + w[2] * test_prob3
    ensemble_test_pred = np.argmax(ensemble_test_prob, axis = 1)

    result = class_encoder.inverse_transform(ensemble_test_pred)

# Submission
if __name__ == '__main__':
    submission = pd.read_csv('E:/PYTHON_FILES/dacon_gravel/sample_submission.csv')
    submission['rock_type'] = result
    submission.to_csv('E:/PYTHON_FILES/dacon_gravel/submission.csv', index = False)
