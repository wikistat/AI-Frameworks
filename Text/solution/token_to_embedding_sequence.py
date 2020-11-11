def tokens_to_embedding_sequences(array_token, model):
    array_embedding_sequences = []
    for tokens in tqdm(array_token):
        embedding_sequence = []
        for token in tokens[:Ns]:
             embedding_sequence.append(model[token])
        array_embedding_sequences.append(embedding_sequence)
    X = pad_sequences(array_embedding_sequences)
    return X