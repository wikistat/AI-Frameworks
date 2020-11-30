user_id_train = rating_train.user_id.values
item_id_train = rating_train.item_id.values
model.compile(optimizer='adam', loss='mse')
history = model.fit(x=[user_id_train, item_id_train], y=rating_train.rating,
                    batch_size=64, epochs=10, validation_split=0.1,
                    shuffle=True)