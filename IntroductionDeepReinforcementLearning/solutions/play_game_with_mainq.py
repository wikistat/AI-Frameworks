state = env.reset()
frames = []
num_step=0
done=False
while not done and num_step < dqn.max_num_step:
    action=np.argmax(dqn.main_qn.model.predict(np.expand_dims(state, axis=0)),axis=1)[0]
    next_state, reward, done = env.step(action)
    frames.append(next_state)
    state=next_state
    num_step+=1
HTML(plot_animation(frames).to_html5_video())