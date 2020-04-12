import joblib
import pandas as pd
import numpy as np
import torch
from torch import nn, optim


X_train = joblib.load('ch08/X_train.joblib')
y_train = joblib.load('ch08/y_train.joblib')
X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

X_test = joblib.load('ch08/X_test.joblib')
y_test = joblib.load('ch08/y_test.joblib')
X_test = torch.from_numpy(X_test.astype(np.float32)).clone()
y_test = torch.from_numpy(y_test.astype(np.int64)).clone()

X = X_train[0:4]
y = y_train[0:4]

net = nn.Linear(X.size()[1], 4)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

losses = []

for epoc in range(100):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss)

_, y_pred_train = torch.max(net(X), 1)
print((y_pred_train == y).sum().item() / len(y))

_, y_pred_test = torch.max(net(X_test), 1)
print((y_pred_test == y_test).sum().item() / len(y_test))
