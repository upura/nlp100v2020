import pandas as pd
from torchtext import data


def load_concat_save(category):
    train_en = pd.read_table(f'ch10/kftt-data-1.0/data/tok/kyoto-{category}.en', header=None)
    train_ja = pd.read_table(f'ch10/kftt-data-1.0/data/tok/kyoto-{category}.ja', header=None, names=[0, 1])
    train_ja.loc[~train_ja[1].isnull(), 0] = train_ja.loc[~train_ja[1].isnull()][0] + ' ' + train_ja.loc[~train_ja[1].isnull()][1]
    train = pd.concat([train_en, train_ja[0]], axis=1)
    train.to_csv(f'ch10/kftt-data-1.0/data/tok/kyoto-{category}.txt', sep='\t', index=False, header=None)


prepare_dataset = False
prepare_list = ['train', 'dev', 'test']
if prepare_dataset:
    for pl in prepare_list:
        load_concat_save(pl)

EN = data.Field(sequential=True, lower=True, batch_first=True)
JA = data.Field(sequential=True, lower=True, batch_first=True)

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='ch10/kftt-data-1.0/data/tok',
    train='kyoto-train.txt',
    validation='kyoto-dev.txt',
    test='kyoto-test.txt',
    format='tsv',
    fields=[('EN', EN), ('JA', JA)])

EN.build_vocab(train_data, min_freq=2)
JA.build_vocab(train_data, min_freq=2)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=128)

batch = next(iter(train_iterator))
print(batch.EN)
print(batch.JA)
