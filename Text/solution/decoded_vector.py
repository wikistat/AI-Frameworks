i_test = 50
print("\nOriginal Sentences:\n%s"%X[i_test])
def decode_sequence(x, int_to_char_dic):
    seq = []
    for i in np.where(x)[1]:
        seq.append(int_to_char_dic[i])
    return "".join(seq)
print("\nDecoded input vector::\n%s"%decode_sequence(X_vec[i_test], int_to_char))
print("\nDecoded output vector::\n%s"%decode_sequence(Y_vec[i_test], int_to_char))
