import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


X_train = joblib.load('ch08/X_train.joblib')
y_train = joblib.load('ch08/y_train.joblib')
X_train = torch.from_numpy(X_train.astype(np.float32)).clone()
y_train = torch.from_numpy(y_train.astype(np.float32)).clone()
w = torch.randn(300, requires_grad=True)

x1 = X_train[0]
X_14 = X_train[0:4]

m = nn.Softmax(dim=-1)
y1 = m(torch.dot(x1, w))
print(y1)
Y = m(torch.mv(X_14, w))
print(Y)
