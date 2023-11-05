import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm


def cosSim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def culcCosSim(row):
    global model
    w1v = model[row['Word 1']]
    w2v = model[row['Word 2']]
    return cosSim(w1v, w2v)


tqdm.pandas()
model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)
df = pd.read_csv('ch07/wordsim353/combined.csv')
df['cosSim'] = df.progress_apply(culcCosSim, axis=1)

print(df[['Human (mean)', 'cosSim']].corr(method='spearman'))
