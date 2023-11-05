import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

X_train = joblib.load('ch08/X_train.joblib')
y_train = joblib.load('ch08/y_train.joblib')
X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
y_train = torch.from_numpy(y_train.astype(np.int64)).clone()

X_valid = joblib.load('ch08/X_valid.joblib')
y_valid = joblib.load('ch08/y_valid.joblib')
X_valid = torch.from_numpy(X_valid.astype(np.float32)).clone()
y_valid = torch.from_numpy(y_valid.astype(np.int64)).clone()

X_test = joblib.load('ch08/X_test.joblib')
y_test = joblib.load('ch08/y_test.joblib')
X_test = torch.from_numpy(X_test.astype(np.float32)).clone()
y_test = torch.from_numpy(y_test.astype(np.int64)).clone()

X = X_train[0:4]
y = y_train[0:4]

net = nn.Linear(X.size()[1], 4)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train_losses = []
valid_losses = []
train_accs = []
valid_accs = []

for epoc in range(100):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

    train_losses.append(loss)
    valid_losses.append(loss_fn(net(X_valid), y_valid))

    _, y_pred_train = torch.max(net(X), 1)
    train_accs.append((y_pred_train == y).sum().item() / len(y))
    _, y_pred_valid = torch.max(net(X_valid), 1)
    valid_accs.append((y_pred_valid == y_valid).sum().item() / len(y_valid))

plt.plot(train_losses, label='train loss')
plt.plot(valid_losses, label='valid loss')
plt.legend()
plt.show()

plt.plot(train_accs, label='train acc')
plt.plot(valid_accs, label='valid acc')
plt.legend()
plt.show()
