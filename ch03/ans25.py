import re
import pandas as pd


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ukText = df.query('title=="イギリス"')['text'].values
ls, fg = [], False
template = '基礎情報'
p1 = re.compile('\{\{' + template)
p2 = re.compile('\}\}')
p3 = re.compile('\|')
p4 = re.compile('<ref(\s|>).+?(</ref>|$)')
for l in ukText[0].split('\n'):
    if fg:
        ml = [p2.match(l), p3.match(l)]
        if ml[0]:
            break
        if ml[1]:
            ls.append(p4.sub('', l.strip()))
    if p1.match(l):
        fg = True
p = re.compile('\|(.+?)\s=\s(.+)')
ans = {m.group(1): m.group(2) for m in [p.match(c) for c in ls]}
print(ans)
