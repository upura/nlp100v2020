import joblib
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def text2id(text):
    return [word2token[word] for word in text.split()]


X_train = pd.read_table('ch06/train.txt', header=None)
use_cols = ['TITLE', 'CATEGORY']
X_train.columns = use_cols

d = defaultdict(int)
for sentence in X_train['TITLE']:
    for word in sentence.split():
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

word2token = dict(zip(words, idx))
print(X_train['TITLE'].apply(text2id))
