import numpy as np
from gensim.models import KeyedVectors


def cosSim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)
result = model.most_similar(positive=['United_States'])

for i in range(10):
    print("{}: {:.4f}".format(*result[i]))
