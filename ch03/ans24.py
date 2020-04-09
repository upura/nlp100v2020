import re
import pandas as pd


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ukText = df.query('title=="イギリス"')['text'].values
for file in re.findall(r'\[\[(ファイル|File):([^]|]+?)(\|.*?)+\]\]', ukText[0]):
    print(file[1])
