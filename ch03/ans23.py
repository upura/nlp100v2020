import re
import pandas as pd


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ukText = df.query('title=="イギリス"')['text'].values
for section in re.findall(r'(=+)([^=]+)\1\n', ukText[0]):
    print('{}\t{}'.format(section[1].strip(), len(section[0]) - 1))
