import joblib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm


def culcSwem(row):
    global model
    swem = [model[w] if w in model.key_to_index else np.zeros(shape=(model.vector_size,)) for w in row['TITLE'].split()]
    swem = np.mean(np.array(swem), axis=0)
    return swem


X_train = pd.read_table('ch06/train.txt', header=None)
X_valid = pd.read_table('ch06/valid.txt', header=None)
X_test = pd.read_table('ch06/test.txt', header=None)
use_cols = ['TITLE', 'CATEGORY']
n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)
X_train.columns = use_cols
X_valid.columns = use_cols
X_test.columns = use_cols

data = pd.concat([X_train, X_valid, X_test]).reset_index(drop=True)

tqdm.pandas()
model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)
swemVec = data.progress_apply(culcSwem, axis=1)

X_train = np.array(list(swemVec.values)[:n_train])
X_valid = np.array(list(swemVec.values)[n_train:n_train + n_valid])
X_test = np.array(list(swemVec.values)[n_train + n_valid:])
joblib.dump(X_train, 'ch08/X_train.joblib')
joblib.dump(X_valid, 'ch08/X_valid.joblib')
joblib.dump(X_test, 'ch08/X_test.joblib')

y_data = data['CATEGORY']

y_train = y_data.values[:n_train]
y_valid = y_data.values[n_train:n_train + n_valid]
y_test = y_data.values[n_train + n_valid:]

joblib.dump(y_train, 'ch08/y_train.joblib')
joblib.dump(y_valid, 'ch08/y_valid.joblib')
joblib.dump(y_test, 'ch08/y_test.joblib')
