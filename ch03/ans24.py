import re
import pandas as pd


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ukText = df.query('title=="イギリス"')['text'].values[0]
for file in re.findall(r'\[\[(ファイル|File):([^]|]+?)(\|.*?)+\]\]', ukText):
    print(file[1])
