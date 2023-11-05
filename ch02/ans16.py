import sys

import pandas as pd

if len(sys.argv) == 1:
    print('Set arg n, like "python ch02/ans15.py 5"')
else:
    n = int(sys.argv[1])
    df = pd.read_csv('ch02/popular-names.txt', sep='\t', header=None)
    nrow = -(-len(df) // n)

    for i in range(n):
        df.loc[nrow * i:nrow * (i + 1)].to_csv(f'ch02/ans16_{i}', sep='\t', index=False, header=None)
