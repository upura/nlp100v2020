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


filename = 'ch05/neko.txt.cabocha'
with open(filename, mode='rt', encoding='utf-8') as f:
    blockList = f.read().split('EOS\n')
blockList = list(filter(lambda x: x != '', blockList))
blockList = [parseCabocha(block) for block in blockList]

with open('ch05/ans47.txt', mode='w') as f:
    for b in blockList:
        for i, m in enumerate(b):
            if 'サ変接続' in [s.pos1 for s in m.morphs] and 'を' in [s.surface for s in m.morphs] and i + 1 < len(b) and b[i + 1].morphs[0].pos == '動詞':
                text = ''.join([s.surface for s in m.morphs]) + b[i + 1].morphs[0].base
                if len(m.srcs) > 0:
                    preMorphs = [b[int(s)].morphs for s in m.srcs]
                    preMorphsFiltered = [list(filter(lambda x: '助詞' in x.pos, pm)) for pm in preMorphs]
                    preSurface = [[p.surface for p in pm] for pm in preMorphsFiltered]
                    preSurface = list(filter(lambda x: x != [], preSurface))
                    preSurface = [p[0] for p in preSurface]
                    preText = list(filter(lambda x: '助詞' in [p.pos for p in x], preMorphs))
                    preText = [''.join([p.surface for p in pt]) for pt in preText]
                    if len(preSurface) > 0:
                        f.writelines('\t'.join([text, ' '.join(preSurface), ' '.join(preText)]))
                        f.write('\n')
