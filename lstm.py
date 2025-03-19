+*In[5]:*+
[source, ipython3]
----
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Define an improved LSTM model architecture with attention
class ArSLAttentionLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=2, bidirectional=True, dropout_rate=0.5):
        super(ArSLAttentionLSTM, self).__init__()
        
        # Use a pre-trained CNN for better feature extraction
        self.feature_extractor = models.resnet18(pretrained=True)
        # Remove the last layer (classification layer)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        
        # Calculate feature size after CNN
        # ResNet18 outputs [batch, 512, 7, 7] for 224x224 input
        feature_dim = 512 * 7 * 7
        
        # Reshape dimensions
        self.reshape_size = 512
        self.seq_length = 49  # 7*7
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.reshape_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layers with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features with CNN
        # x shape: [batch, 3, 224, 224]
        x = self.feature_extractor(x)  # [batch, 512, 7, 7]
        
        # Reshape for LSTM: [batch, channels, height, width] -> [batch, seq_len, features]
        batch_size = x.size(0)
        x = x.view(batch_size, self.reshape_size, -1).permute(0, 2, 1)  # [batch, 49, 512]
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1).unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_size*2]
        
        # Apply dropout
        x = self.dropout(context_vector)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Function to create transforms with more augmentation
def get_transforms(train=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),  # Larger resize for cropping
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

# Modified training function with learning rate scheduler
def train_model(model, train_loader, criterion, optimizer, scheduler, 
                num_epochs=60, device=torch.device("cuda"), model_path="models/improved_arsl_model.pth"):
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Track best metrics
    best_train_acc = 0.0
    train_losses = []
    train_accs = []
    
    print("Starting training with learning rate scheduling...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total, 'lr': optimizer.param_groups[0]['lr']})
        
        # Step the scheduler
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model if improved
        if epoch_acc > best_train_acc:
            best_train_acc = epoch_acc
            print(f"New best training accuracy: {best_train_acc:.2f}% - Saving model...")
            
            # Save model checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_acc': epoch_acc,
                'class_names': train_loader.dataset.classes,
                'metrics': {
                    'best_train_acc': best_train_acc,
                    'train_losses': train_losses,
                    'train_accs': train_accs
                }
            }
            
            torch.save(checkpoint, model_path)
        
        # Also save at regular intervals
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"models/checkpoint_arsl_epoch_{epoch+1}.pth"
            print(f"Saving checkpoint at epoch {epoch+1}...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_acc': epoch_acc
            }, checkpoint_path)
    
    print(f"Training completed. Best training accuracy: {best_train_acc:.2f}%")
    return train_losses, train_accs

# Improved evaluation function with more detailed metrics
def evaluate_model(model, test_loader, criterion, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities for ROC analysis
    total_loss = 0
    correct = 0
    total = 0
    
    print("Generating predictions...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
            # Calculate accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    test_loss = total_loss / len(test_loader.dataset)
    accuracy = 100. * correct / total
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Generate classification report
    print("\nGenerating classification report...")
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Print report as table
    print("\n===== Classification Report =====")
    report_table = pd.DataFrame(report_dict).transpose()
    print(report_table)
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Identify most confused pairs
    n_classes = len(class_names)
    error_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                error_pairs.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': cm[i, j],
                    'error_rate': cm[i, j] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                })
    
    error_df = pd.DataFrame(error_pairs)
    top_errors = error_df.sort_values('count', ascending=False).head(10)
    
    print("\n===== Top 10 Confusion Pairs =====")
    print(top_errors)
    
    return accuracy, report_df, cm, all_probs

# Main execution function
def main(train_dir, test_dir, model_path="models/improved_arsl_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # Create class weights for imbalanced data
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    model = ArSLAttentionLSTM(num_classes=num_classes).to(device)
    
    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=60,
        pct_start=0.1
    )
    
    # Set mode (train or test)
    mode = "train_and_test"  # Options: "train", "test", "train_and_test"
    
    if mode in ["train", "train_and_test"]:
        # Train model
        train_losses, train_accs = train_model(
            model, train_loader, criterion, optimizer, scheduler,
            num_epochs=60, device=device, model_path=model_path
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        
        plt.tight_layout()
        plt.savefig('improved_training_history.png', dpi=300)
        print("Training history saved to 'improved_training_history.png'")
    
    if mode in ["test", "train_and_test"]:
        # Load the best model for testing
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate the model
        accuracy, report_df, cm, probs = evaluate_model(
            model, test_loader, criterion, device, train_dataset.classes
        )
        
        # Save report
        report_df.to_csv('improved_classification_report.csv')
        print("Classification report saved to 'improved_classification_report.csv'")
        
        # Plot class-wise metrics
        metrics_df = report_df.iloc[:-3]  # Exclude avg rows
        metrics_df = metrics_df.sort_values(by=['f1-score'], ascending=False)
        
        plt.figure(figsize=(15, 8))
        x = np.arange(len(metrics_df))
        width = 0.25
        
        plt.bar(x - width, metrics_df['precision'], width, label='Precision')
        plt.bar(x, metrics_df['recall'], width, label='Recall')
        plt.bar(x + width, metrics_df['f1-score'], width, label='F1-score')
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Class-wise Performance Metrics')
        plt.xticks(x, metrics_df.index, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig('improved_class_performance.png', dpi=300)
        print("Class performance metrics saved to 'improved_class_performance.png'")

if __name__ == "__main__":
    train_dir = "C:/Users/Fatima/Downloads/train"
    test_dir = "C:/Users/Fatima/Downloads/test"
    main(train_dir, test_dir)
----


+*Out[5]:*+
----
Using device: cuda
Number of classes: 32

Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to C:\Users\Fatima/.cache\torch\hub\checkpoints\resnet18-f37072fd.pth
  0%|          | 0.00/44.7M [00:00<?, ?B/s]
Starting training with learning rate scheduling...

Epoch 1/60: 100%|████████████████████████████████████| 1537/1537 [03:25<00:00,  7.50it/s, loss=1.02, acc=43.3, lr=4e-5]

Epoch 1/60 - Train Loss: 2.0867, Train Acc: 43.26%, LR: 0.000040
New best training accuracy: 43.26% - Saving model...

Epoch 2/60: 100%|███████████████████████████████████| 1537/1537 [03:24<00:00,  7.51it/s, loss=0.502, acc=87.1, lr=4e-5]

Epoch 2/60 - Train Loss: 0.5077, Train Acc: 87.13%, LR: 0.000040
New best training accuracy: 87.13% - Saving model...

Epoch 3/60: 100%|████████████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=1.47, acc=92.5, lr=4e-5]

Epoch 3/60 - Train Loss: 0.2935, Train Acc: 92.48%, LR: 0.000040
New best training accuracy: 92.48% - Saving model...

Epoch 4/60: 100%|████████████████████████████████████| 1537/1537 [03:21<00:00,  7.62it/s, loss=0.58, acc=94.4, lr=4e-5]

Epoch 4/60 - Train Loss: 0.2133, Train Acc: 94.42%, LR: 0.000040
New best training accuracy: 94.42% - Saving model...

Epoch 5/60: 100%|█████████████████████████████████████| 1537/1537 [03:21<00:00,  7.62it/s, loss=0.205, acc=95, lr=4e-5]

Epoch 5/60 - Train Loss: 0.1886, Train Acc: 94.99%, LR: 0.000040
New best training accuracy: 94.99% - Saving model...

Epoch 6/60: 100%|██████████████████████████████████| 1537/1537 [03:21<00:00,  7.62it/s, loss=0.0135, acc=95.7, lr=4e-5]

Epoch 6/60 - Train Loss: 0.1664, Train Acc: 95.67%, LR: 0.000040
New best training accuracy: 95.67% - Saving model...

Epoch 7/60: 100%|███████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.372, acc=96.3, lr=4e-5]

Epoch 7/60 - Train Loss: 0.1413, Train Acc: 96.33%, LR: 0.000040
New best training accuracy: 96.33% - Saving model...

Epoch 8/60: 100%|██████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.0862, acc=96.5, lr=4e-5]

Epoch 8/60 - Train Loss: 0.1398, Train Acc: 96.49%, LR: 0.000040
New best training accuracy: 96.49% - Saving model...

Epoch 9/60: 100%|█████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.00153, acc=96.9, lr=4e-5]

Epoch 9/60 - Train Loss: 0.1212, Train Acc: 96.89%, LR: 0.000040
New best training accuracy: 96.89% - Saving model...

Epoch 10/60: 100%|███████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=1.05, acc=97.2, lr=4e-5]

Epoch 10/60 - Train Loss: 0.1124, Train Acc: 97.22%, LR: 0.000040
New best training accuracy: 97.22% - Saving model...
Saving checkpoint at epoch 10...

Epoch 11/60: 100%|████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.00316, acc=97.4, lr=4e-5]

Epoch 11/60 - Train Loss: 0.1070, Train Acc: 97.36%, LR: 0.000040
New best training accuracy: 97.36% - Saving model...

Epoch 12/60: 100%|█████████████████████████████████| 1537/1537 [03:22<00:00,  7.61it/s, loss=0.0295, acc=97.5, lr=4e-5]

Epoch 12/60 - Train Loss: 0.1032, Train Acc: 97.48%, LR: 0.000040
New best training accuracy: 97.48% - Saving model...

Epoch 13/60: 100%|████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.00208, acc=97.7, lr=4e-5]

Epoch 13/60 - Train Loss: 0.0936, Train Acc: 97.75%, LR: 0.000040
New best training accuracy: 97.75% - Saving model...

Epoch 14/60: 100%|████████████████████████████████| 1537/1537 [03:22<00:00,  7.61it/s, loss=0.00283, acc=97.8, lr=4e-5]

Epoch 14/60 - Train Loss: 0.0947, Train Acc: 97.76%, LR: 0.000040
New best training accuracy: 97.76% - Saving model...

Epoch 15/60: 100%|█████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.0382, acc=97.9, lr=4e-5]

Epoch 15/60 - Train Loss: 0.0872, Train Acc: 97.93%, LR: 0.000040
New best training accuracy: 97.93% - Saving model...

Epoch 16/60: 100%|██████████████████████████████████| 1537/1537 [03:21<00:00,  7.61it/s, loss=0.658, acc=98.2, lr=4e-5]

Epoch 16/60 - Train Loss: 0.0799, Train Acc: 98.19%, LR: 0.000040
New best training accuracy: 98.19% - Saving model...

Epoch 17/60: 100%|██████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.743, acc=98.1, lr=4e-5]

Epoch 17/60 - Train Loss: 0.0804, Train Acc: 98.13%, LR: 0.000040

Epoch 18/60: 100%|█████████████████████████████████| 1537/1537 [03:23<00:00,  7.55it/s, loss=0.0398, acc=98.3, lr=4e-5]

Epoch 18/60 - Train Loss: 0.0768, Train Acc: 98.26%, LR: 0.000040
New best training accuracy: 98.26% - Saving model...

Epoch 19/60: 100%|████████████████████████████████| 1537/1537 [03:23<00:00,  7.56it/s, loss=0.00266, acc=98.3, lr=4e-5]

Epoch 19/60 - Train Loss: 0.0720, Train Acc: 98.34%, LR: 0.000040
New best training accuracy: 98.34% - Saving model...

Epoch 20/60: 100%|████████████████████████████████| 1537/1537 [03:23<00:00,  7.55it/s, loss=0.00183, acc=98.5, lr=4e-5]

Epoch 20/60 - Train Loss: 0.0681, Train Acc: 98.46%, LR: 0.000040
New best training accuracy: 98.46% - Saving model...
Saving checkpoint at epoch 20...

Epoch 21/60: 100%|████████████████████████████████| 1537/1537 [03:23<00:00,  7.56it/s, loss=0.00135, acc=98.5, lr=4e-5]

Epoch 21/60 - Train Loss: 0.0657, Train Acc: 98.54%, LR: 0.000040
New best training accuracy: 98.54% - Saving model...

Epoch 22/60: 100%|████████████████████████████████| 1537/1537 [03:23<00:00,  7.55it/s, loss=0.00655, acc=98.5, lr=4e-5]

Epoch 22/60 - Train Loss: 0.0670, Train Acc: 98.46%, LR: 0.000040

Epoch 23/60: 100%|████████████████████████████████| 1537/1537 [03:23<00:00,  7.56it/s, loss=0.00569, acc=98.6, lr=4e-5]

Epoch 23/60 - Train Loss: 0.0634, Train Acc: 98.62%, LR: 0.000040
New best training accuracy: 98.62% - Saving model...

Epoch 24/60: 100%|██████████████████████████████████| 1537/1537 [03:21<00:00,  7.62it/s, loss=0.443, acc=98.6, lr=4e-5]

Epoch 24/60 - Train Loss: 0.0655, Train Acc: 98.58%, LR: 0.000040

Epoch 25/60: 100%|██████████████████████████████████| 1537/1537 [03:21<00:00,  7.62it/s, loss=0.859, acc=98.7, lr=4e-5]

Epoch 25/60 - Train Loss: 0.0594, Train Acc: 98.72%, LR: 0.000040
New best training accuracy: 98.72% - Saving model...

Epoch 26/60: 100%|██████████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.496, acc=98.7, lr=4e-5]

Epoch 26/60 - Train Loss: 0.0608, Train Acc: 98.73%, LR: 0.000040
New best training accuracy: 98.73% - Saving model...

Epoch 27/60: 100%|██████████████████████████████████| 1537/1537 [03:23<00:00,  7.54it/s, loss=0.758, acc=98.8, lr=4e-5]

Epoch 27/60 - Train Loss: 0.0583, Train Acc: 98.76%, LR: 0.000040
New best training accuracy: 98.76% - Saving model...

Epoch 28/60: 100%|████████████████████████████████| 1537/1537 [03:23<00:00,  7.56it/s, loss=0.00335, acc=98.8, lr=4e-5]

Epoch 28/60 - Train Loss: 0.0599, Train Acc: 98.76%, LR: 0.000040
New best training accuracy: 98.76% - Saving model...

Epoch 29/60: 100%|███████████████████████████████| 1537/1537 [03:23<00:00,  7.56it/s, loss=0.000863, acc=98.7, lr=4e-5]

Epoch 29/60 - Train Loss: 0.0582, Train Acc: 98.70%, LR: 0.000040

Epoch 30/60: 100%|████████████████████████████████| 1537/1537 [03:22<00:00,  7.61it/s, loss=0.00161, acc=98.9, lr=4e-5]

Epoch 30/60 - Train Loss: 0.0516, Train Acc: 98.89%, LR: 0.000040
New best training accuracy: 98.89% - Saving model...
Saving checkpoint at epoch 30...

Epoch 31/60: 100%|████████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.00974, acc=98.9, lr=4e-5]

Epoch 31/60 - Train Loss: 0.0529, Train Acc: 98.89%, LR: 0.000040

Epoch 32/60: 100%|█████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.0103, acc=98.9, lr=4e-5]

Epoch 32/60 - Train Loss: 0.0537, Train Acc: 98.88%, LR: 0.000040

Epoch 33/60: 100%|██████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.00437, acc=99, lr=4e-5]

Epoch 33/60 - Train Loss: 0.0489, Train Acc: 98.97%, LR: 0.000040
New best training accuracy: 98.97% - Saving model...

Epoch 34/60: 100%|████████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.00206, acc=98.9, lr=4e-5]

Epoch 34/60 - Train Loss: 0.0485, Train Acc: 98.94%, LR: 0.000040

Epoch 35/60: 100%|██████████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.469, acc=99.1, lr=4e-5]

Epoch 35/60 - Train Loss: 0.0454, Train Acc: 99.08%, LR: 0.000040
New best training accuracy: 99.08% - Saving model...

Epoch 36/60: 100%|██████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.00224, acc=99, lr=4e-5]

Epoch 36/60 - Train Loss: 0.0477, Train Acc: 99.05%, LR: 0.000040

Epoch 37/60: 100%|██████████████████████████████████| 1537/1537 [03:23<00:00,  7.56it/s, loss=0.00751, acc=99, lr=4e-5]

Epoch 37/60 - Train Loss: 0.0465, Train Acc: 99.03%, LR: 0.000040

Epoch 38/60: 100%|█████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000452, acc=99, lr=4e-5]

Epoch 38/60 - Train Loss: 0.0464, Train Acc: 99.05%, LR: 0.000040

Epoch 39/60: 100%|███████████████████████████████| 1537/1537 [03:22<00:00,  7.58it/s, loss=0.000661, acc=99.1, lr=4e-5]

Epoch 39/60 - Train Loss: 0.0445, Train Acc: 99.10%, LR: 0.000040
New best training accuracy: 99.10% - Saving model...

Epoch 40/60: 100%|███████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=1.81, acc=99.1, lr=4e-5]

Epoch 40/60 - Train Loss: 0.0490, Train Acc: 99.06%, LR: 0.000040
Saving checkpoint at epoch 40...

Epoch 41/60: 100%|███████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000524, acc=99.1, lr=4e-5]

Epoch 41/60 - Train Loss: 0.0422, Train Acc: 99.14%, LR: 0.000040
New best training accuracy: 99.14% - Saving model...

Epoch 42/60: 100%|█████████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.0759, acc=99.1, lr=4e-5]

Epoch 42/60 - Train Loss: 0.0440, Train Acc: 99.11%, LR: 0.000040

Epoch 43/60: 100%|████████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.00205, acc=99.2, lr=4e-5]

Epoch 43/60 - Train Loss: 0.0421, Train Acc: 99.16%, LR: 0.000040
New best training accuracy: 99.16% - Saving model...

Epoch 44/60: 100%|█████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.00773, acc=99.2, lr=4.01e-5]

Epoch 44/60 - Train Loss: 0.0387, Train Acc: 99.18%, LR: 0.000040
New best training accuracy: 99.18% - Saving model...

Epoch 45/60: 100%|█████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.00061, acc=99.2, lr=4.01e-5]

Epoch 45/60 - Train Loss: 0.0373, Train Acc: 99.20%, LR: 0.000040
New best training accuracy: 99.20% - Saving model...

Epoch 46/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000261, acc=99.2, lr=4.01e-5]

Epoch 46/60 - Train Loss: 0.0393, Train Acc: 99.17%, LR: 0.000040

Epoch 47/60: 100%|█████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=9.76e-5, acc=99.2, lr=4.01e-5]

Epoch 47/60 - Train Loss: 0.0406, Train Acc: 99.24%, LR: 0.000040
New best training accuracy: 99.24% - Saving model...

Epoch 48/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000568, acc=99.2, lr=4.01e-5]

Epoch 48/60 - Train Loss: 0.0382, Train Acc: 99.23%, LR: 0.000040

Epoch 49/60: 100%|█████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.00119, acc=99.3, lr=4.01e-5]

Epoch 49/60 - Train Loss: 0.0379, Train Acc: 99.26%, LR: 0.000040
New best training accuracy: 99.26% - Saving model...

Epoch 50/60: 100%|██████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.0166, acc=99.2, lr=4.01e-5]

Epoch 50/60 - Train Loss: 0.0394, Train Acc: 99.20%, LR: 0.000040
Saving checkpoint at epoch 50...

Epoch 51/60: 100%|█████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.00585, acc=99.2, lr=4.01e-5]

Epoch 51/60 - Train Loss: 0.0386, Train Acc: 99.23%, LR: 0.000040

Epoch 52/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.000639, acc=99.2, lr=4.01e-5]

Epoch 52/60 - Train Loss: 0.0407, Train Acc: 99.18%, LR: 0.000040

Epoch 53/60: 100%|███████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.011, acc=99.3, lr=4.01e-5]

Epoch 53/60 - Train Loss: 0.0355, Train Acc: 99.29%, LR: 0.000040
New best training accuracy: 99.29% - Saving model...

Epoch 54/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000408, acc=99.3, lr=4.01e-5]

Epoch 54/60 - Train Loss: 0.0357, Train Acc: 99.30%, LR: 0.000040
New best training accuracy: 99.30% - Saving model...

Epoch 55/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000887, acc=99.3, lr=4.01e-5]

Epoch 55/60 - Train Loss: 0.0372, Train Acc: 99.26%, LR: 0.000040

Epoch 56/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.60it/s, loss=0.000163, acc=99.3, lr=4.01e-5]

Epoch 56/60 - Train Loss: 0.0348, Train Acc: 99.26%, LR: 0.000040

Epoch 57/60: 100%|███████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.331, acc=99.3, lr=4.01e-5]

Epoch 57/60 - Train Loss: 0.0320, Train Acc: 99.31%, LR: 0.000040
New best training accuracy: 99.31% - Saving model...

Epoch 58/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000706, acc=99.3, lr=4.01e-5]

Epoch 58/60 - Train Loss: 0.0358, Train Acc: 99.31%, LR: 0.000040

Epoch 59/60: 100%|███████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.444, acc=99.3, lr=4.01e-5]

Epoch 59/60 - Train Loss: 0.0344, Train Acc: 99.30%, LR: 0.000040

Epoch 60/60: 100%|████████████████████████████| 1537/1537 [03:22<00:00,  7.59it/s, loss=0.000311, acc=99.4, lr=4.01e-5]

Epoch 60/60 - Train Loss: 0.0301, Train Acc: 99.38%, LR: 0.000040
New best training accuracy: 99.38% - Saving model...
Saving checkpoint at epoch 60...
Training completed. Best training accuracy: 99.38%
Training history saved to 'improved_training_history.png'
Loading model from models/improved_arsl_model.pth
Generating predictions...

100%|████████████████████████████████████████████████████████████████████████████████| 828/828 [00:32<00:00, 25.41it/s]

Test Loss: 0.1284
Test Accuracy: 98.19%

Generating classification report...

===== Classification Report =====
              precision    recall  f1-score       support
ain            0.988787  0.994872  0.991820    975.000000
al             0.992126  0.998415  0.995261    631.000000
aleff          0.992366  0.988593  0.990476    263.000000
bb             0.991416  0.969570  0.980371    953.000000
dal            0.975420  0.966667  0.971024    780.000000
dha            0.992604  0.958571  0.975291    700.000000
dhad           0.986721  0.985714  0.986217    980.000000
fa             0.954680  0.972892  0.963700    996.000000
gaaf           0.981108  0.946537  0.963513    823.000000
ghain          0.992972  0.994970  0.993970    994.000000
ha             0.977221  0.986207  0.981693    870.000000
haa            0.981432  0.963542  0.972405    768.000000
jeem           0.973778  0.979616  0.976689    834.000000
kaaf           0.982049  0.988310  0.985169    941.000000
khaa           0.988610  0.984127  0.986364    882.000000
la             0.989437  0.984813  0.987119    856.000000
laam           0.955988  0.995736  0.975457    938.000000
meem           0.992840  0.970828  0.981711    857.000000
nun            0.967890  0.981395  0.974596    860.000000
ra             0.964591  0.989975  0.977118    798.000000
saad           0.975771  0.985539  0.980631    899.000000
seen           0.986715  0.995128  0.990904    821.000000
sheen          0.997661  0.995333  0.996495    857.000000
ta             0.972350  0.980256  0.976287    861.000000
taa            0.995470  0.977753  0.986532    899.000000
thaa           0.988263  0.971165  0.979639    867.000000
thal           0.984211  0.980341  0.982272    763.000000
toot           0.980482  0.991870  0.986143    861.000000
waw            0.975765  0.990933  0.983290    772.000000
ya             0.992282  0.990099  0.991189    909.000000
yaa            0.994307  0.961468  0.977612    545.000000
zay            0.974700  0.991870  0.983210    738.000000
accuracy       0.981918  0.981918  0.981918      0.981918
macro avg      0.982500  0.981659  0.982005  26491.000000
weighted avg   0.982053  0.981918  0.981913  26491.000000

Generating confusion matrix...

===== Top 10 Confusion Pairs =====
     true predicted  count  error_rate
37   gaaf        fa     25    0.030377
25    dha        ta     22    0.031429
157   yaa      laam     15    0.027523
128  thaa      kaaf     14    0.016148
53    haa      jeem     13    0.016927
35     fa      saad     11    0.011044
38   gaaf        ha     10    0.012151
12     bb        ra     10    0.010493
72     la       nun      9    0.010514
10     bb      laam      9    0.009444
Classification report saved to 'improved_classification_report.csv'
Class performance metrics saved to 'improved_class_performance.png'

![png](output_0_126.png)

![png](output_0_127.png)

![png](output_0_128.png)
----


+*In[ ]:*+
[source, ipython3]
----

----
