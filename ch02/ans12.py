import pandas as pd


df = pd.read_csv('ch02/popular-names.txt', sep='\t', header=None)
df[0].to_csv('ch02/col1.txt', index=False, header=None)
df[1].to_csv('ch02/col2.txt', index=False, header=None)
