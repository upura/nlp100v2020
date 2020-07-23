def parse_mecab(block):
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


def extract_noun_noun(block):
    res = []
    tmp = []
    for b in block:
        if b['pos'] == '名詞':
            tmp.append(b['surface'])
        elif len(tmp) >= 2:
            res.append(''.join(tmp))
            tmp = []
        else:
            tmp = []
    return res


filename = 'ch04/neko.txt.mecab'
with open(filename, mode='rt', encoding='utf-8') as f:
    blocks = f.read().split('EOS\n')
blocks = list(filter(lambda x: x != '', blocks))
blocks = [parse_mecab(block) for block in blocks]
ans = [extract_noun_noun(block) for block in blocks]
print(ans)
