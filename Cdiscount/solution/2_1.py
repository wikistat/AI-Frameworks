ir = 0
print("Liste des tokens racinisé de la première ligne : " + train_array[0])
rw = data_train_OHE.getrow(ir)
pd.DataFrame([(v, vocabulary[v], k)  for k,v in zip(rw.data,rw.indices)], columns=["indices","token","weight"])