t0 = time()
df_with_label.select("label", "protocol_type").groupBy("label", "protocol_type").count().show()
tt = time() - t0

print ("Requete executee en {} secondes".format(round(tt,3)))