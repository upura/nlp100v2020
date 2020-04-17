import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


X_train = pd.read_table('ch06/train.feature.txt', header=None)
X_test = pd.read_table('ch06/test.feature.txt', header=None)
y_train = pd.read_table('ch06/train.txt', header=None)[1]
y_test = pd.read_table('ch06/test.txt', header=None)[1]

clf = joblib.load('ch06/model.joblib')

print(f'train acc: {accuracy_score(y_train, clf.predict(X_train))}')
print(f'test acc: {accuracy_score(y_test, clf.predict(X_test))}')
