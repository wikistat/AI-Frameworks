x_pred = np.zeros((1, Ns+1, Nv))
print("step 0")
x_pred[0,0,char_to_int["START"]] =1
letter_start = "I" 
i_letter_start = char_to_int[letter_start]
x_pred[0,1,i_letter_start] =1
x_pred_str = decode_sequence(x_pred[0], int_to_char)
print(x_pred_str)

for i in range(1,Ns):
    x_tensor = convert_to_tensor(x_pred[:,:i+1,:])
    ix = np.argmax(model.predict(x_tensor)[0][-1,:])
    x_pred[0,i+1,ix] = 1
x_pred_str=decode_sequence(x_pred[0], int_to_char)
print(x_pred_str)