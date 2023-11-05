import torch
from torch import optim
from torchtext import data
from tqdm import tqdm
from transformers import BertForSequenceClassification


def eval_net(model, data_loader, device='cpu'):
    model.eval()
    ys = []
    ypreds = []
    for x, y, _ in data_loader:
        with torch.no_grad():
            loss, logit = model(input_ids=x, labels=y)
            _, y_pred = torch.max(logit, 1)
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    print(f'test acc: {(ys == ypreds).sum().item() / len(ys)}')
    return


TEXT = data.Field(sequential=True, lower=True, batch_first=True)
LABELS = data.Field(sequential=False, batch_first=True, use_vocab=False)

train, val, test = data.TabularDataset.splits(
    path='ch06', train='train2.txt',
    validation='valid2.txt', test='test2.txt', format='tsv',
    fields=[('TEXT', TEXT), ('LABEL', LABELS)])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), batch_sizes=(64, 64, 64), device=device, repeat=False, sort=False)

TEXT.build_vocab(train, min_freq=2)
LABELS.build_vocab(train)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in tqdm(range(10)):
    losses = []
    model.train()
    for batch in train_iter:
        x, y = batch.TEXT, batch.LABEL
        loss, logit = model(input_ids=x, labels=y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, y_pred_train = torch.max(logit, 1)
    eval_net(model, test_iter, device)
