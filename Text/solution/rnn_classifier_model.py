model = km.Sequential()
model.add(kl.LSTM(units=256 ,activation="relu", input_shape=(28, 300)))
model.add(kl.Dense(256))
model.add(kl.Activation("relu"))
model.add(kl.Dense(N_label))
model.add(kl.Activation("softmax"))
model.summary()

epochs = 500
batch_size=256
history = model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=[X_valid, Y_valid])