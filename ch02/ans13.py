import pandas as pd


c1 = pd.read_csv('ch02/col1.txt', header=None)
c2 = pd.read_csv('ch02/col2.txt', header=None)

df = pd.concat([c1, c2], axis=1)
df.to_csv('ch02/ans13.txt', sep='\t', index=False, header=None)
