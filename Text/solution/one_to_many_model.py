model = km.Sequential()
model.add(kl.SimpleRNN(units=10 ,activation="relu", input_shape=(None, 1), return_sequences=True))
model.add(kl.TimeDistributed(kl.Dense(1)))

epochs = 1000
batch_size=32
model.compile(loss="mse", optimizer="adam")
model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)