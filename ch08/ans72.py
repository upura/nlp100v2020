import joblib
import numpy as np
import torch
import torch.nn as nn


X_train = joblib.load('ch08/X_train.joblib')
y_train = joblib.load('ch08/y_train.joblib')
X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

X = X_train[0:4]
y = y_train[0:4]

net = nn.Linear(X.size()[1], 4)
y_pred = net(X)
print(y_pred)
print(y)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y)
print(loss)
