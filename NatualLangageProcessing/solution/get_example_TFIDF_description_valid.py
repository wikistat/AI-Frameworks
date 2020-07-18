ir = 2
rw = data_valid_TFIDF.getrow(ir)
print("Liste des tokens racinisé de la première ligne : " + data_valid.Description_cleaned.values[ir])
pd.DataFrame([(ind, vocabulary[ind], vec.idf_[ind], w/vec.idf_[ind], w)  for w,ind in zip(rw.data,rw.indices)], columns=["indices","token","idf","tf","weight"])
