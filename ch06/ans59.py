import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


X_train = pd.read_table('ch06/train.feature.txt', header=None)
X_valid = pd.read_table('ch06/valid.feature.txt', header=None)
X_test = pd.read_table('ch06/test.feature.txt', header=None)
y_train = pd.read_table('ch06/train.txt', header=None)[1]
y_valid = pd.read_table('ch06/valid.txt', header=None)[1]
y_test = pd.read_table('ch06/test.txt', header=None)[1]

test_acc = []

C_candidate = [0.1, 1.0, 10, 100]
for c in C_candidate:
    clf = LogisticRegression(penalty='l2', solver='sag', random_state=0, C=c)
    clf.fit(X_train, y_train)
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))


max_depth_candidate = [2, 4, 8, 16]
for m in max_depth_candidate:
    clf = RandomForestClassifier(max_depth=m, random_state=0)
    clf.fit(X_train, y_train)
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

bestIndex = test_acc.index(max(test_acc))
if bestIndex < 4:
    bestAlg = 'LogisticRegression'
    bestParam = f'C={C_candidate[bestIndex]}'
else:
    bestAlg = 'RandomForestClassifier'
    bestParam = f'max_depth={max_depth_candidate[bestIndex - 4]}'

print(bestAlg, bestParam)
