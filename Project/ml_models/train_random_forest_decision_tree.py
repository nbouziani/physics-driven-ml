import torch
from torch.utils.data import DataLoader
from Project.ml_models.ml_models import NN, randomForest, decisionTree
from firedrake import *
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from Project.data_loader.Dataset import Data
import numpy as np
import os


def parseDataset(isTensor=False):
    X = np.load("../../data/datasets/linear_elasticity/X.npy", allow_pickle=True)
    y = np.load("../../data/datasets/linear_elasticity/y.npy", allow_pickle=True)
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

