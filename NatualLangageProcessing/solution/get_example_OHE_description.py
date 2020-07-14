ir = 47
print("Liste des tokens racinisé de la première ligne : " + data_train.Description_cleaned.values[ir])
rw = data_train_OHE.getrow(ir)
pd.DataFrame([(v, vocabulary[v], k)  for k,v in zip(rw.data,rw.indices)], columns=["indices","token","weight"])