token_positif = ["femme","roi"]
token_negatif = ["homme"]

token_positif_stem = [stemmer.stem(token) for token in token_positif]
token_negatif_stem = [stemmer.stem(token) for token in token_negatif]


columns = ["CBOW", "skip-gram", "online"]
df_ = []
msw_cbow = model_cbow.wv.most_similar(positive=token_positif_stem, negative=token_negatif_stem)
df_.append([msw_cbow[k][0] for k in range(10)])
msw_sg = model_sg.wv.most_similar(positive=token_positif_stem, negative=token_negatif_stem)
df_.append([msw_sg[k][0] for k in range(10)])
msw_online = model_online.wv.most_similar(positive=token_positif, negative=token_negatif)
df_.append([msw_online[k][0] for k in range(10)])
pd.DataFrame(np.array(df_).T, columns=columns)