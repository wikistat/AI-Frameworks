user_id_input = kl.Input(shape=[1], name='user')
item_id_input = kl.Input(shape=[1], name='item')

embedding_size = 30
user_embedding = kl.Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = kl.Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = kl.Flatten()(user_embedding)
item_vecs = kl.Flatten()(item_embedding)

input_vecs = kl.Concatenate()([user_vecs, item_vecs])
input_vecs = kl.Dropout(0.5)(input_vecs)

x = kl.Dense(64, activation='relu')(input_vecs)
y = kl.Dense(1)(x)

model = km.Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(optimizer='adam', loss='mae')
model.summary()