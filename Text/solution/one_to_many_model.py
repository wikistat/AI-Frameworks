model = km.Sequential()
model.add(kl.SimpleRNN(units=10 ,activation="relu", input_shape=(None, 1), return_sequences=True))
model.add(kl.TimeDistributed(kl.Dense(1)))