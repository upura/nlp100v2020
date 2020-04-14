import re
import joblib
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW


def cleanText(text):
    remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;]|<.*?>")
    shift_marks_regex = re.compile("([?!])")
    # !?以外の記号の削除
    text = remove_marks_regex.sub("", text)
    # !?と単語の間にスペースを挿入
    text = shift_marks_regex.sub(r" \1 ", text)
    return text


def list2tensor(token_idxes, max_len=20, padding=True):
    if len(token_idxes) > max_len:
        token_idxes = token_idxes[:max_len]
    n_tokens = len(token_idxes)
    if padding:
        token_idxes = token_idxes + [0] * (max_len - len(token_idxes))
    return torch.tensor(token_idxes, dtype=torch.int64), n_tokens


class TITLEDataset(Dataset):
    def __init__(self, section='train'):
        X_train = pd.read_table(f'ch06/{section}.txt', header=None)
        use_cols = ['TITLE', 'CATEGORY']
        X_train.columns = use_cols

        d = defaultdict(int)
        for text in X_train['TITLE']:
            text = cleanText(text)
            for word in text.split():
                d[word] += 1
        dc = sorted(d.items(), key=lambda x: x[1], reverse=True)

        words = []
        idx = []
        for i, a in enumerate(dc, 1):
            words.append(a[0])
            if a[1] < 2:
                idx.append(0)
            else:
                idx.append(i)

        self.word2token = dict(zip(words, idx))
        self.data = (X_train['TITLE'].apply(lambda x: list2tensor(
            [self.word2token[word] if word in self.word2token.keys() else 0 for word in cleanText(x).split()])))

        y_train = pd.read_table(f'ch06/{section}.txt', header=None)[1].map({'b': 0, 'e': 1, 't': 2, 'm': 3}).values
        self.labels = y_train

    @property
    def vocab_size(self):
        return len(self.word2token)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data, n_tokens = self.data[idx]
        label = self.labels[idx]
        return data, label, n_tokens


def eval_net(net, data_loader, device='cpu'):
    net.eval()
    ys = []
    ypreds = []
    for x, y, _ in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            loss, logit = net(input_ids=x, labels=y)
            _, y_pred = torch.max(logit, 1)
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    print(f'test acc: {(ys == ypreds).sum().item() / len(ys)}')
    return


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 640
train_data = TITLEDataset(section='train')
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, num_workers=4)
test_data = TITLEDataset(section='test')
test_loader = DataLoader(test_data, batch_size=batch_size,
                         shuffle=False, num_workers=4)
net = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
net = net.to(device)
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in tqdm(range(10)):
    losses = []
    net.train()
    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)
        loss, logit = net(input_ids=x, labels=y)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _, y_pred_train = torch.max(logit, 1)
    eval_net(net, test_loader, device)
