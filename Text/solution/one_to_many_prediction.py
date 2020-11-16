def predict_function(x):
    x_test=np.array(x).reshape(1,1,1)
    y1 = model.predict(x_test)
    y2 = model.predict(np.hstack((x_test,y1)))
    y3 = model.predict(np.hstack((x_test,y2)))
    return y3

x=10
y=predict_function(x)
print("Input scalar : %d. Output sequences: [%.3f, %.3f, %.3f]" %(x, y[0][0],y[0][1][0],y[0][2][0]) )