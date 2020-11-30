idx = 1027
X = np.expand_dims(item_embeddings[idx],axis=0)
distX = sm.pairwise_distances(X, item_embeddings, metric="cosine")[0]
print("Top 10 items similar to movies %s" %str(id_item_to_title[idx]))
mostSimilarItem = pd.DataFrame([[id_item_to_title[x], distX[x],x] for x in distX.argsort()[:10]])
mostSimilarItem