from collections import defaultdict
import matplotlib.pyplot as plt
import japanize_matplotlib


def parseMecab(block):
    res = []
    for line in block.split('\n'):
        if line == '':
            return res
        (surface, attr) = line.split('\t')
        attr = attr.split(',')
        lineDict = {
            'surface': surface,
            'base': attr[6],
            'pos': attr[0],
            'pos1': attr[1]
        }
        res.append(lineDict)


def extract(block):
    return [b['base'] for b in block]


filename = 'ch04/neko.txt.mecab'
with open(filename, mode='rt', encoding='utf-8') as f:
    blockList = f.read().split('EOS\n')
blockList = list(filter(lambda x: x != '', blockList))
blockList = [parseMecab(block) for block in blockList]
wordList = [extract(block) for block in blockList]
wordList = list(filter(lambda x: '猫' in x, wordList))
d = defaultdict(int)
for word in wordList:
    for w in word:
        if w != '猫':
            d[w] += 1
ans = sorted(d.items(), key=lambda x: x[1], reverse=True)[:10]
labels = [a[0] for a in ans]
values = [a[1] for a in ans]
plt.figure(figsize=(8, 8))
plt.barh(labels, values)
plt.show()
