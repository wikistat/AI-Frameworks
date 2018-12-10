tokens = ["coque","pour"]
#tokens = ["homme"]
#tokens = ["femme"]

tokens_stem = [stemmer.stem(token) for token in tokens]


columns = ["CBOW", "skip-gram", "online"]
df_ = []
msw_cbow = model_cbow.predict_output_word(tokens_stem)
df_.append([msw_cbow[k][0] for k in range(10)])
msw_sg = model_sg.predict_output_word(tokens_stem)
df_.append([msw_sg[k][0] for k in range(10)])
msw_online = model_online.predict_output_word(tokens)
df_.append([msw_online[k][0] for k in range(10)])
pd.DataFrame(np.array(df_).T, columns=columns)