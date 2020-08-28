import pandas as pd
from word_embedding import WordEmbedding
from solution.clean import CleanText

ct = CleanText()
data = pd.read_csv("data/Categorie_original.zip",sep=";").fillna("")
ct.clean_df_column(data, "Description", "Description_cleaned")
array_token = [line.split(" ") for line in data["Description_cleaned"].values]
print(len(array_token))

features_dimension = 300
min_count = 1
window = 5
hs = 0
negative = 10

we_sg = WordEmbedding(word_embedding_type = "word2vec",
                      args = dict(sentences = array_token, sg=1, hs=hs, negative=negative, min_count=min_count, size=features_dimension, window = window, iter=15))
model_sg, training_time_sg = we_sg.train()
print("Model Skip-gram trained in %.2f minutes"%(training_time_sg/60))


we_cbow = WordEmbedding(word_embedding_type = "word2vec",
                      args = dict(sentences = array_token, sg=0, hs=hs, negative=negative, min_count=min_count, size=features_dimension, window = window, iter=15))
model_cbow, training_time_cbow = we_cbow.train()
print("Model CBOW trained in %.2f minutes"%(training_time_cbow/60))

model_sg.save("data/full_model_sg")
model_cbow.save("data/full_model_cbow")
