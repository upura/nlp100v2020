import torch
from torch import nn, optim
import torch.nn.functional as F
from torchtext import data
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback
from torch.utils.data import DataLoader
from torchtext.data import Iterator
from gensim.models import KeyedVectors


class BucketIteratorWrapper(DataLoader):
    __initialized__ = False

    def __init__(self, iterator: Iterator):
        self.batch_size = iterator.batch_size
        self.num_workers = 1
        self.collate_fn = None
        self.pin_memory = False
        self.drop_last = False
        self.timeout = 0
        self.worker_init_fn = None
        self.sampler = iterator
        self.batch_sampler = iterator
        self.__initialized__ = True

    def __iter__(self):
        return map(lambda batch: {
            'features': batch.TEXT,
            'targets': batch.LABEL,
        }, self.batch_sampler.__iter__())

    def __len__(self):
        return len(self.batch_sampler)


class CNN(nn.Module):

    def __init__(self, output_dim, kernel_num, kernel_sizes=[3, 4, 5], dropout=0.5, static=False):
        super(CNN, self).__init__()

        model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)
        weights = torch.FloatTensor(model.vectors)
        self.embed = nn.Embedding.from_pretrained(weights)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num, (k, self.embed.weight.shape[1])) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, output_dim)
        self.static = static

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)

        if self.static:
            x = x.detach()

        x = x.unsqueeze(1)
        x = x.float()
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


TEXT = data.Field(sequential=True, lower=True, batch_first=True)
LABELS = data.Field(sequential=False, batch_first=True, use_vocab=False)

train, val, test = data.TabularDataset.splits(
    path='ch06', train='train2.txt',
    validation='valid2.txt', test='test2.txt', format='tsv',
    fields=[('TEXT', TEXT), ('LABEL', LABELS)])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test), batch_sizes=(64, 64, 64), device=device, repeat=False, sort=False)

train_loader = BucketIteratorWrapper(train_iter)
valid_loader = BucketIteratorWrapper(val_iter)
loaders = {"train": train_loader, "valid": valid_loader}

TEXT.build_vocab(train, min_freq=2)
LABELS.build_vocab(train)
model = CNN(output_dim=4, kernel_num=3, kernel_sizes=[3, 4, 5], dropout=0.2)

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
