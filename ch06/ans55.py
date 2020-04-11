import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


X_train = pd.read_table('ch06/train.feature.txt', header=None)
X_test = pd.read_table('ch06/test.feature.txt', header=None)
y_train = pd.read_table('ch06/train.txt', header=None)[1]
y_test = pd.read_table('ch06/test.txt', header=None)[1]

clf = joblib.load('ch06/model.joblib')

print(f'train confusion matrix:\n {confusion_matrix(y_train, clf.predict(X_train))}')
print(f'test confusion matrix:\n {confusion_matrix(y_test, clf.predict(X_test))}')
