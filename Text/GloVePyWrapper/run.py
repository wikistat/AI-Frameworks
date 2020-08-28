import glove_pywrapper

CORPUS = "text8"
glove = glove_pywrapper.GloveWrapper(CORPUS, "text8")
#prepare vocabulary count
glove.vocab_count()
#prepare co-occurrence matrix
glove.cooccur()
#reshuffle
glove.shuffle()
#glove train
glove.glove()