import re
import pandas as pd


def remove_stress(dc):
    r = re.compile("'+")
    return {k: r.sub('', v) for k, v in dc.items()}


def remove_inner_links(dc):
    r = re.compile('\[\[(.+\||)(.+?)\]\]')
    return {k: r.sub(r'\2', v) for k, v in dc.items()}


def removeMk(v):
    r1 = re.compile("'+")
    r2 = re.compile('\[\[(.+\||)(.+?)\]\]')
    r3 = re.compile('\{\{(.+\||)(.+?)\}\}')
    r4 = re.compile('<\s*?/*?\s*?br\s*?/*?\s*>')
    v = r1.sub('', v)
    v = r2.sub(r'\2', v)
    v = r3.sub(r'\2', v)
    v = r4.sub('', v)
    return v


df = pd.read_json('ch03/jawiki-country.json.gz', lines=True)
ukText = df.query('title=="イギリス"')['text'].values[0]

ls, fg = [], False
template = '基礎情報'
p1 = re.compile('\{\{' + template)
p2 = re.compile('\}\}')
p3 = re.compile('\|')
p4 = re.compile('<ref(\s|>).+?(</ref>|$)')
for l in ukText.split('\n'):
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
r = re.compile('\[\[(.+\||)(.+?)\]\]')
ans = {k: r.sub(r'\2', removeMk(v)) for k, v in ans.items()}
print(remove_inner_links(remove_stress(ans)))
