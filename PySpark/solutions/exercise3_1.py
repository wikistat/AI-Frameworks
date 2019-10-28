t0 = time()
df_with_label.select("label").groupBy("label").count().show()
tt = time() - t0

print ("Requete executee en {} secondes".format(round(tt,3)))