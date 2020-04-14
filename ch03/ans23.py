import re
import pandas as pd


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ukText = df.query('title=="イギリス"')['text'].values[0]
for section in re.findall(r'(=+)([^=]+)\1\n', ukText):
    print(f'{section[1].strip()}\t{len(section[0]) - 1}')
