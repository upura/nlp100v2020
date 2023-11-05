import torch
from torch import nn
from torchtext import data


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

    def forward(self, x, h0=None):
        x = self.emb(x)
        x, h = self.lstm(x, h0)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


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
model = RNN(len(TEXT.vocab.stoi) + 1, num_layers=2, output_size=4)

for epoch in range(1):
    model.train()
    for batch in train_iter:
        x, y = batch.TEXT, batch.LABEL
        y_pred = model(x)
        print(y_pred)
        print(y_pred.shape)
