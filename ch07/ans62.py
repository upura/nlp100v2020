from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('ch07/GoogleNews-vectors-negative300.bin', binary=True)
result = model.most_similar(positive=['United_States'])

for i in range(10):
    print("{}: {:.4f}".format(*result[i]))
