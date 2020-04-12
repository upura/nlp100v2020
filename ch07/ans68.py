import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


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
linkage_result = linkage(X, method='ward', metric='euclidean')
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=countryName)
plt.show()
