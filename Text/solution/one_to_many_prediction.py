def predict_function(x):
    x_test=np.array(x).reshape(1,1,1)
    y1 = model.predict(x_test)
    y2 = model.predict(y1)
    y3 = model.predict(y2)
    return [y1,y2,y3]

x=10
y=predict_function(x)
print("Input scalar : %d. Output sequences: [%.3f, %.3f, %.3f]" %(x, y[0],y[1],y[2]) )