from cnn import BCNN, BCNN_Red
from data import get_dataloaders
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Calculate class weights using training data
def compute_weights(train_loader):
    train_labels = [label for _, label in train_loader.dataset.samples]  # Extract labels
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
    return class_weights_tensor

def train(model,train_loader, valid_loader, criterion, optimizer, num_epochs, save_dir, train_name, results_csv="results.csv"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print('Running process to device : ',device)

    best_val_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+"/"+train_name, exist_ok=True)

    # Initialize results list for storing metrics
    results = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()

        # Training phase
        train_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()#error handling
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        
        # Validation phase
        avg_val_loss, val_accuracy = validate(model, valid_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        # Log results
        results.append({
            "Epoch": epoch + 1,
            "Train Loss": avg_train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": val_accuracy
        })

        
        print(f'Epoch [{epoch+1}/{num_epochs}]: '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f},'
              f'Learning Rate: {current_lr}')
        
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, train_name, f"{train_name}_best.pt"))
            print(f"Best model saved with Validation Loss: {avg_val_loss:.4f}")

    # Save last model at the end of training
    torch.save(model.state_dict(), os.path.join(save_dir, train_name, f"{train_name}_last.pt"))
    print("Last model saved.")

    # Save results to a CSV file
    # Full path for the results CSV
    results_path = os.path.join(save_dir, train_name)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_path, results_csv), index=False)
    print(f"Results saved to {results_csv}")
    
def validate(model, valid_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  # No gradient calculation for validation
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_val_loss = total_val_loss / len(valid_loader)
    val_accuracy = total_correct / total_samples
    return avg_val_loss, val_accuracy

def test(test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure device consistency
    net.to(device)
    run_test =input('Run on test data ? [Y/n]')

    if ((run_test == 'Y') or (run_test == 'y')):
        print('Testing trained model on the test set...')
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    else:
        print("Closing...")

if __name__ == '__main__':

    net = BCNN_Red()

    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=32, num_workers=8)

    # Compute weights
    class_weights_tensor = compute_weights(train_loader)

    # Use weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 60

    save_dir = 'models'
    train_name = 'train_BCNN_R_Leaky_60'

    train_folder = os.path.join(save_dir, train_name)
    last_model_path = os.path.join(train_folder, f"{train_name}_last.pt")

    if os.path.exists(train_folder):
        if os.path.exists(last_model_path):
            print(f"Found existing model: {last_model_path}. Loading for continued training...")
            net.load_state_dict(torch.load(last_model_path))  # Load the last saved model

    train(net, train_loader, valid_loader, criterion, optimizer, num_epochs, save_dir, train_name)
    test(test_loader)