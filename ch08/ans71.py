import joblib
import numpy as np
import torch
import torch.nn as nn

X_train = joblib.load('ch08/X_train.joblib')
X_train = torch.from_numpy(X_train.astype(np.float32)).clone()

X = X_train[0:4]

net = nn.Sequential(nn.Linear(X.size()[1], 4), nn.Softmax(1))
y_pred = net(X)
print(y_pred)
