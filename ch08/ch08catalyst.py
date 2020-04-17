import joblib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback


def get_Xy(category):
    X = joblib.load(f'ch08/X_{category}.joblib')
    y = joblib.load(f'ch08/y_{category}.joblib')
    X = torch.from_numpy(X.astype(np.float32)).clone()
    y = torch.from_numpy(y.astype(np.int64)).clone()
    return X, y


def get_loader(X, y, batch_size=64, shuffle=True):
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


X_train, y_train = get_Xy('train')
X_valid, y_valid = get_Xy('valid')
X_test, y_test = get_Xy('test')
train_loader = get_loader(X_train, y_train)
valid_loader = get_loader(X_valid, y_valid)
loaders = {"train": train_loader, "valid": valid_loader}

model = nn.Linear(X_train.size()[1], 4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
runner = SupervisedRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logdir",
    callbacks=[AccuracyCallback(num_classes=4, accuracy_args=[1])],
    num_epochs=10,
    verbose=True,
)

test_loader = get_loader(X_test, y_test)
logits = runner.predict_loader(model=model, loader=test_loader, verbose=True)

y_pred = torch.max(torch.from_numpy(logits), dim=1)[1]
print(y_pred[:10])
