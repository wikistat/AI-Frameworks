X = []
Y = []
for i in range(100):
    X.append([i,i+2,i+4])
    Y.append([i+2,i+4,i+6])
X = np.array(X).reshape(100,3,1)
Y = np.array(Y).reshape(100,3,1)