import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import torch.optim as optim
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange


class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model, self).__init__()

        # Define your model architecture
        self.fc1 = nn.Linear(5, 8)  # Input size is 6
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 3)  # Output size is 4

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        return x


def read_dataset():
    X = np.load("../../data/datasets/linear_elasticity/X.npy",
                allow_pickle=True)
    y = np.load("../../data/datasets/linear_elasticity/y.npy",
                allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_cross_validation(model, dataset, k_folds, epochs, batch_size, lr, criterion, device, save_folder):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_losses = []
    val_losses = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Define data subsets for training and validation
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        # Re-initialize model for each fold
        model_fold = model()
        model_fold.to(device)
        optimizer_fold = optim.Adam(model_fold.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        # Training loop for each fold
        for epoch in tqdm(range(epochs)):
            model_fold.train()
            current_train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_fold.zero_grad()
                outputs = model_fold(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_fold.step()
                current_train_loss += loss.item()

            current_val_loss = 0.0
            model_fold.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model_fold(inputs)
                    loss = criterion(outputs, targets)
                    current_val_loss += loss.item()

            train_losses.append(current_train_loss / len(train_loader))
            val_losses.append(current_val_loss / len(val_loader))

            if epoch == epochs - 1:
                print(f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
            # Save model if the validation loss has decreased
            best_val_loss = np.Inf
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                torch.save(model_fold.state_dict(), os.path.join(save_folder, f'best_model.pt'))
    # Plotting after all folds
    plt.figure(figsize=(12, 4))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


X_train, X_test, y_train, y_test = read_dataset()
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# Compute the mean and standard deviation for the training data
X_mean = torch.mean(X_train_tensor, dim=0)
# X_min = torch.min(X_train_tensor, dim=1)
# X_max = torch.max(X_train_tensor, dim=1)
X_std = torch.std(X_train_tensor, dim=0)
y_mean = torch.mean(y_train_tensor, dim=0)
# y_min = torch.min(y_train_tensor, dim=1)
# y_max = torch.max(y_train_tensor, dim=1)
y_std = torch.std(y_train_tensor, dim=0)

# Standardize the training data
X_train_standardized = (X_train_tensor - X_mean) / X_std
y_train_standardized = (y_train_tensor - y_mean) / y_std

# Standardize the test data using the training data's statistics
X_test_standardized = (X_test_tensor - X_mean) / X_std
y_test_standardized = (y_test_tensor - y_mean) / y_std
WEIGHT_DECAY = 0.01  # Regularization strength

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model, criterion and optimizer
model = MLP_model().to(device)
criterion = nn.MSELoss()
# Hyperparameters
EPOCHS = 500
LEARNING_RATE = 0.0003
BATCH_SIZE = 16
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Dataset loading utilities
train_dataset = torch.utils.data.TensorDataset(X_train_standardized, y_train_standardized)
test_dataset = torch.utils.data.TensorDataset(X_test_standardized, y_test_standardized)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
save_folder = './'
# Using the function:
combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
train_cross_validation(MLP_model, combined_dataset, k_folds=3, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                       criterion=criterion, device=device, save_folder=save_folder)
