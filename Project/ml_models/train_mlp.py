from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ml_models import NN


def read_dataset():
    X = np.load("../../data/datasets/linear_elasticity/X.npy",
                allow_pickle=True)
    y = np.load("../../data/datasets/linear_elasticity/y.npy",
                allow_pickle=True)

    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32) \
        , torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)


def train_mlp_model(train_loader, epochs=500, lr=0.0002):
    loss_record = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{500}], Loss: {total_loss / len(X_train)}")
        loss_record.append(total_loss / len(X_train))
    torch.save(model.state_dict(), "best.pth")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 501), loss_record, label='Training Loss')

    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = read_dataset()
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    train_mlp_model(train_loader)

