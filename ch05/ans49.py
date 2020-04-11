class Morph:
    def __init__(self, dc):
        self.surface = dc['surface']
        self.base = dc['base']
        self.pos = dc['pos']
        self.pos1 = dc['pos1']


class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs    # 形態素（Morphオブジェクト）のリスト
        self.dst = dst          # 係り先文節インデックス番号
        self.srcs = []          # 係り元文節インデックス番号のリスト


def parseCabocha(block):
    def checkCreateChunk(tmp):
        if len(tmp) > 0:
            c = Chunk(tmp, dst)
            res.append(c)
            tmp = []
        return tmp

    res = []
    tmp = []
    dst = None
    for line in block.split('\n'):
        if line == '':
            tmp = checkCreateChunk(tmp)
        elif line[0] == '*':
            dst = line.split(' ')[2].rstrip('D')
            tmp = checkCreateChunk(tmp)
        else:
            (surface, attr) = line.split('\t')
            attr = attr.split(',')
            lineDict = {
                'surface': surface,
                'base': attr[6],
                'pos': attr[0],
                'pos1': attr[1]
            }
            tmp.append(Morph(lineDict))

    for i, r in enumerate(res):
        res[int(r.dst)].srcs.append(i)
    return res


def convert(s):
    pl, nl = [], [c for c in s if '名詞' in [m.pos for m in c.morphs]]
    for i in range(len(nl) - 1):
        st1 = [''.join([m.surface if m.pos != '名詞' else 'X' for m in nl[i].morphs])]
        for e in nl[i + 1:]:
            dst, p = nl[i].dst, []
            st2 = [''.join([m.surface if m.pos != '名詞' else 'Y' for m in e.morphs])]
            while int(dst) != -1 and dst != s.index(e):
                p.append(s[int(dst)])
                dst = s[int(dst)].dst
            if len(p) < 1 or p[-1].dst != -1:
                mid = [''.join([m.surface for m in c.morphs if m.pos != '記号']) for c in p]
                pl.append(st1 + mid + ['Y'])
            else:
                mid, dst = [], e.dst
                while not s[int(dst)] in p:
                    mid.append(''.join([m.surface for m in s[int(dst)].morphs if m.pos != '記号']))
                    dst = s[int(dst)].dst
                ed = [''.join([m.surface for m in s[int(dst)].morphs if m.pos != '記号'])]
                pl.append([st1, st2 + mid, ed])
    return pl


filename = 'ch05/neko.txt.cabocha'
with open(filename, mode='rt', encoding='utf-8') as f:
    blockList = f.read().split('EOS\n')
blockList = list(filter(lambda x: x != '', blockList))
blockList = [parseCabocha(block) for block in blockList]

for b in blockList:
    pl = (convert(b))
    for p in pl:
        if isinstance(p[0], str):
            print(' -> '.join(p))
        else:
            print(p[0][0], ' -> '.join(p[1]), p[2][0], sep=' | ')
