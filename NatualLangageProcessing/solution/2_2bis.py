ir = 0
rw = data_valid_TFIDF.getrow(ir)
print("Liste des tokens racinisé de la première ligne : " + valid_array[0])
pd.DataFrame([(ind, vocabulary[ind], vec.idf_[ind], w/vec.idf_[ind], w)  for w,ind in zip(rw.data,rw.indices)], columns=["indices","token","idf","tf","weight"])
