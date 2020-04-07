import pandas as pd


df = pd.read_csv('ch02/popular-names.txt', sep='\t', header=None)
df.to_csv('ch02/ans11.txt', sep=' ', index=False, header=None)
