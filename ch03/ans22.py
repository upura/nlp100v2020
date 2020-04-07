import pandas as pd


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ans = df.query('title=="イギリス"')['category'].values[0]
print(ans[0])
