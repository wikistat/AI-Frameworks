def my_normalize(v):
    norm = sum(v)
    return v/norm*0.99

X_pred = np.zeros((1, Ns+1, Nv))
print("step 0")
X_pred[0,0,char_to_int["START"]] = 1
X_pred_str = decode_sequence(X_pred[0], int_to_char)
print(X_pred_str)

for i in range(197):
    i=i
    predict_step = my_normalize(model.predict(X_pred[:,:i+1,:])[0][-1,:])
    ix = np.argmax(np.random.multinomial(1,predict_step, size=1))
    X_pred[0,i+1,ix] = 1
    X_pred_str = decode_sequence(X_pred[0], int_to_char)
    print(X_pred_str, end="\r")