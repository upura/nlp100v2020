import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

# http://www.fao.org/countryprofiles/iso3list/en/
country = pd.read_table('ch07/countries.tsv')
country = country['Short name'].values

model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)

countryVec = []
countryName = []
for c in country:
    if c in model.vocab:
        countryVec.append(model[c])
        countryName.append(c)

X = np.array(countryVec)
tsne = TSNE(random_state=0, n_iter=15000, metric='cosine')
embs = tsne.fit_transform(X)
plt.scatter(embs[:, 0], embs[:, 1])
plt.show()
