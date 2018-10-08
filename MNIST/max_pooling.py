from keras.layers import MaxPool2D
conv_mp = Sequential([ MaxPool2D(pool_size=(2,2))])

img_in = np.expand_dims(x, 0)
img_out = conv_mp.predict(img_in)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(img_in[0,:,:,0].astype(np.uint8),
           cmap="binary");
ax0.grid(False)

ax1.imshow(img_out[0,:,:,0].astype(np.uint8),
           cmap="binary");
ax1.grid(False)