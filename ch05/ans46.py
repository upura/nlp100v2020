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


def parse_cabocha(block):
    def check_create_chunk(tmp):
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
            tmp = check_create_chunk(tmp)
        elif line[0] == '*':
            dst = line.split(' ')[2].rstrip('D')
            tmp = check_create_chunk(tmp)
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


filename = 'ch05/ai.ja.txt.cabocha'
with open(filename, mode='rt', encoding='utf-8') as f:
    blocks = f.read().split('EOS\n')
blocks = list(filter(lambda x: x != '', blocks))
blocks = [parse_cabocha(block) for block in blocks]

for b in blocks:
    for m in b:
        if len(m.srcs) > 0:
            pre_morphs = [b[int(s)].morphs for s in m.srcs]
            pre_morphs_filtered = [list(filter(lambda x: '助詞' in x.pos, pm)) for pm in pre_morphs]
            pre_surface = [[p.surface for p in pm] for pm in pre_morphs_filtered]
            pre_surface = list(filter(lambda x: x != [], pre_surface))
            pre_surface = [p[0] for p in pre_surface]
            post_base = [mo.base for mo in m.morphs]
            post_pos = [mo.pos for mo in m.morphs]
            if len(pre_surface) > 0 and '動詞' in post_pos:
                pre_text = list(filter(lambda x: '助詞' in [p.pos for p in x], pre_morphs))
                pre_text = [''.join([p.surface for p in pt]) for pt in pre_text]
                print(post_base[0], ' '.join(pre_surface), ' '.join(pre_text), sep='\t')
