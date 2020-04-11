from sklearn.model_selection import train_test_split
import pandas as pd


newsCorpora = pd.read_table('ch06/NewsAggregatorDataset/newsCorpora.csv', header=None)
newsCorpora.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
newsCorpora = newsCorpora[newsCorpora['PUBLISHER'].isin(
    ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])].sample(frac=1, random_state=0)

X = newsCorpora[['TITLE', 'CATEGORY']]
y = newsCorpora['CATEGORY']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=0)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=0)

X_train.to_csv('ch06/train.txt', sep='\t', index=False, header=None)
X_valid.to_csv('ch06/valid.txt', sep='\t', index=False, header=None)
X_test.to_csv('ch06/test.txt', sep='\t', index=False, header=None)
