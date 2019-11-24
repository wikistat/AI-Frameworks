done = False
num_step = 0
sum_rewards = []
state = env.reset()
frames = []
while not done and num_step < dqn.max_num_step:
    action = np.argmax(dqn.main_qn.model.predict(np.array([state])),axis=1)[0]
    next_state, reward, done = env.step(action)
    sum_rewards.append(reward)
    frames.append(next_state)
    num_step += 1
    state=next_state
HTML(plot_animation(frames).to_html5_video())