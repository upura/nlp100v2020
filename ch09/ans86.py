# ref: https://www.shoeisha.co.jp/book/detail/9784798157184
import re
from collections import defaultdict

import joblib
import pandas as pd
import torch
from gensim.models import KeyedVectors
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


class CNN(nn.Module):
    def __init__(self, num_embeddings,
                 embedding_dim=300,
                 hidden_size=300,
                 output_size=1,
                 kernel_size=3):
        super().__init__()
        # self.emb = nn.Embedding(num_embeddings, embedding_dim,
        #                         padding_idx=0)
        model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)
        weights = torch.FloatTensor(model.vectors)
        self.emb = nn.Embedding.from_pretrained(weights)
        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=hidden_size,
                      kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(20 - kernel_size + 1))
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.emb(x)
        content_out = self.content_conv(x.permute(0, 2, 1))
        reshaped = content_out.view(content_out.size(0), -1)
        x = self.linear(reshaped)
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


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 640
    train_data = TITLEDataset(section='train')
    train_loader = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    net = CNN(train_data.vocab_size + 1, output_size=4)
    net = net.to(device)

    for epoch in tqdm(range(10)):
        net.train()
        for x, y, nt in train_loader:
            x = x.to(device)
            y = y.to(device)
            nt = nt.to(device)
            y_pred = net(x)
