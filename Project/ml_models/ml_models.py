import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# Define the architecture of the ML model
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # Define your model architecture
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def randomForest(X_train, y_train, X_test, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    # joblib.dump(rfr, '../../Project/saved_models/random_forest.pkl')
    # R2 score between [0, 1]. The higher score the better performance
    print("Random forest evaluation(R2 score): {:.2f}".format(rfr.score(X_test, y_test)))


def decisionTree(X_train, y_train, X_test, y_test):
    dtr = DecisionTreeRegressor()
    dtr.fit(X_train, y_train)
    # joblib.dump(dtr, '../../Project/saved_models/decision_tree.pkl')
    # R2 score between [0, 1]. The higher score the better performance
    print("Decision Tree evaluation(R2 score): {:.2f}".format(dtr.score(X_test, y_test)))
