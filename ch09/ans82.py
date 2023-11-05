# ref: https://www.shoeisha.co.jp/book/detail/9784798157184
import re
from collections import defaultdict

import joblib
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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


class RNN(nn.Module):
    def __init__(self, num_embeddings,
                 embedding_dim=50,
                 hidden_size=50,
                 output_size=1,
                 num_layers=1,
                 dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim,
                                padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, n_tokens=None):
        # IDをEmbeddingで多次元のベクトルに変換する
        # xは(batch_size, step_size)
        # -> (batch_size, step_size, embedding_dim)
        x = self.emb(x)
        # 初期状態h0と共にRNNにxを渡す
        # xは(batch_size, step_size, embedding_dim)
        # -> (batch_size, step_size, hidden_dim)
        x, h = self.lstm(x, h0)
        # 最後のステップのみ取り出す
        # xは(batch_size, step_size, hidden_dim)
        # -> (batch_size, 1)
        if n_tokens is not None:
            # 入力のもともとの長さがある場合はそれを使用する
            x = x[list(range(len(x))), n_tokens - 1, :]
        else:
            # なければ単純に最後を使用する
            x = x[:, -1, :]
        # 取り出した最後のステップを線形層に入れる
        x = self.linear(x)
        # 余分な次元を削除する
        # (batch_size, 1) -> (batch_size, )
        # x = x.squeeze()
        return x


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

        y_train = pd.read_table(f'ch06/{section}.txt', header=None)[1].values
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
    for x, y, nt in data_loader:
        with torch.no_grad():
            y_pred = net(x, n_tokens=nt)
            print(f'test loss: {loss_fn(y_pred, y.long()).item()}')
            _, y_pred = torch.max(y_pred, 1)
            ys.append(y)
            ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    print(f'test acc: {(ys == ypreds).sum().item() / len(ys)}')
    return


if __name__ == "__main__":
    train_data = TITLEDataset(section='train')
    train_loader = DataLoader(train_data, batch_size=len(train_data),
                            shuffle=True, num_workers=4)
    test_data = TITLEDataset(section='test')
    test_loader = DataLoader(test_data, batch_size=len(test_data),
                            shuffle=False, num_workers=4)

    net = RNN(train_data.vocab_size + 1, num_layers=2, output_size=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in tqdm(range(10)):
        losses = []
        net.train()
        for x, y, nt in train_loader:
            y_pred = net(x, n_tokens=nt)
            loss = loss_fn(y_pred, y.long())
            net.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            _, y_pred_train = torch.max(y_pred, 1)
            print(f'train loss: {loss.item()}')
            print(f'train acc: {(y_pred_train == y).sum().item() / len(y)}')
        eval_net(net, test_loader)
