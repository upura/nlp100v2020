from torchtext import data


TEXT = data.Field(sequential=True, lower=True, batch_first=True)
LABELS = data.Field(sequential=False, batch_first=True, use_vocab=False)

train, valid, test = data.TabularDataset.splits(
    path='ch06', train='train2.txt',
    validation='valid2.txt', test='test2.txt', format='tsv',
    fields=[('text', TEXT), ('labels', LABELS)])

TEXT.build_vocab(train, min_freq=2)
print(TEXT.vocab.stoi)
