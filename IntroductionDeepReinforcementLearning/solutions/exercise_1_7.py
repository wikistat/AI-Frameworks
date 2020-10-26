frames = []

env.reset()
observation, reward, done, _ = env.step(env.action_space.sample())
reward_sum = 0
while not(done):
    img = env.render(mode = "rgb_array")
    env.close()
    frames.append(img)
    reward_sum += reward
    p_left = pg.model_predict.predict(np.expand_dims(observation,axis=0))
    action = 0 if  p_left>0.5 else 1
    observation, reward, done, _ = env.step(action)
plt.close()
HTML(plot_animation(frames).to_html5_video())