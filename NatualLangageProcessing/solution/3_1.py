i_test = 50
print(X[50])
def decode_sequence(x, int_to_char_dic):
    seq = []
    for i in np.where(x)[1]:
        seq.append(int_to_char_dic[i])
    return "".join(seq)
decode_sequence(X_vec[50], int_to_char)