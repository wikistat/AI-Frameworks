def encode_input_output_sequence(x_descriptions, length_sequence, size_vocab, char_to_int_dic):
    # Get the number of description in x.
    n = x_descriptions.shape[0]

    # Set the dimensions of the output encoded matrices fill with zero.
    # the length_sequence is actually length_sequences
    x_vec = np.zeros((n, length_sequence + 1, size_vocab))
    y_vec = np.zeros((n, length_sequence + 1, size_vocab))

    # Let's now fill the matrices with one at the location of each characters position

    # First let's fill each input sequences with the START position at the begining of the encoded sequences
    x_vec[:, 0, char_to_int["START"]] = 1
    # and the output sequences with the END position at the end of the encoded sequences
    y_vec[:, -1, char_to_int["END"]] = 1
    # Now let's iterate over all x_descriptions
    for ix, x in tqdm(enumerate(x_descriptions)):
        # And over each character of the description
        for ic, c in enumerate(x):
            # For each character `c` we set one at his position in the vocabulary.
            c_int = char_to_int_dic[c]
            x_vec[ix, ic + 1, c_int] = 1
    # The y-vec matrices is the same than the x matrix with one offset
    y_vec[:, :-1, :] = x_vec[:, 1:, :]
    return x_vec, y_vec