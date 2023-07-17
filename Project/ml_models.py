import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import *


# Define the architecture of the ML model
class ElasticModel(nn.Module):
    def __init__(self):
        super(ElasticModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


def randomForest(X_train, y_train, X_test, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    joblib.dump(rfr, 'random_forest.pkl')
    # R2 score between [0, 1]. The higher score the better performance
    print("Random forest evaluation(R2 score): {:.2f}".format(rfr.score(X_test, y_test)))


def decisionTree(X_train, y_train, X_test, y_test):
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    joblib.dump(dtr, 'decision_tree.pkl')
    # R2 score between [0, 1]. The higher score the better performance
    print("Decision Tree evaluation(R2 score): {:.2f}".format(dtr.score(X_test, y_test)))
