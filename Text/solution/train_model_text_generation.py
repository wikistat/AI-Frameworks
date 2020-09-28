nb_hidden = 32
model = km.Sequential()
model.add(kl.LSTM(nb_hidden, input_shape=(None, Nv), return_sequences=True))
model.add(kl.TimeDistributed(kl.Dense(Nv)))
model.add(kl.Activation('softmax'))
model.summary()

epochs = 20
batch_size= 128
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
model.fit(X_vec, Y_vec, epochs=epochs, batch_size=batch_size)
model.save("data/generate_model.h5")