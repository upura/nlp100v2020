import joblib
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt


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

X = X_train
y = y_train
X = X.to('cuda:0')
y = y.to('cuda:0')
ds = TensorDataset(X, y)

net = nn.Sequential(
    nn.Linear(X.size()[1], 100),
    nn.PReLU(),
    nn.BatchNorm1d(100),
    nn.Linear(100, 25),
    nn.PReLU(),
    nn.BatchNorm1d(25),
    nn.Linear(25, 4)
)
net = net.to('cuda:0')
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

batchSize = [64]

for bs in batchSize:
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    for epoc in tqdm(range(100)):
        train_running_loss = 0.0
        valid_running_loss = 0.0

        for xx, yy in loader:
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()
            valid_running_loss += loss_fn(net(X_valid), y_valid).item()

        joblib.dump(net.state_dict(), f'ch08/state_dict_{epoc}.joblib')

        train_losses.append(train_running_loss)
        valid_losses.append(valid_running_loss)

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
