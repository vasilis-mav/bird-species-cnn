# data.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=16, num_workers=8, pin_memory=True, img_size=(224, 224)):
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define the data transformations
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    valid_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # Load datasets
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
    valid_dataset = datasets.ImageFolder(root='data/valid', transform=valid_transform)
    test_dataset = datasets.ImageFolder(root='data/test', transform=valid_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for testing
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, valid_loader, test_loader

