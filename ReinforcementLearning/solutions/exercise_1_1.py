obs = env.reset()
while True:
    obs, reward, done, info = env.step(0)
    if done:
        break
img = env.render(mode = "rgb_array")
env.close()
plt.imshow(img)
axs = plt.axis("off")
