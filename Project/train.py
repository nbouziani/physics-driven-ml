import torch
from torch.utils.data import DataLoader

from model import generate_dataset_by_linear_elastic_forward_model
from ml_models import ElasticModel, randomForest, decisionTree
from firedrake import *
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from Dataset import Data
import numpy as np


def dataToCsv():
    num_samples = 500
    E = Constant(2.1e11)  # Young's modulus in Pa
    nu = Constant(0.3)  # Poisson's ratio
    X, y = generate_dataset_by_linear_elastic_forward_model(num_samples, E, nu)
    result = []
    for i in range(X.shape[0]):
        line = np.concatenate((X[i], y[i]))
        result.append(line)
    data = pd.DataFrame(result)
    data.to_csv("data.csv", index=False)


def parseDataset(isTensor=False):
    data = pd.read_csv('data.csv')
    X = data[['0', '1', '2', '3']].values
    y = data[['4', '5', '6', '7']].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if isTensor:
        return torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_train).float(), \
               torch.from_numpy(y_test).float()
    else:
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # dataToCsv() # This is used to generate dataset

    # -------- Train random forest and decision tree -----------
    X_train, X_test, y_train, y_test = parseDataset()  # parse dataset and split train and test set
    randomForest(X_train, y_train, X_test, y_test)
    decisionTree(X_train, y_train, X_test, y_test)

    # -------- Train MLP -----------
    X_train, X_test, y_train, y_test = parseDataset(True)  # parse dataset and split train and test set

    train_dataset = Data(X_train, y_train)
    test_dataset = Data(X_test, y_test)
    train_set = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    test_set = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)
    criterion = nn.MSELoss()
    model = ElasticModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # Training loop
    for epoch in range(50):
        total_loss = 0
        cnt = 0
        for i, data in enumerate(train_set):
            cnt += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{50}], Loss: {total_loss / cnt}")
