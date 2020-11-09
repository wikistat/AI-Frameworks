state = env.reset()
frames = []
num_step=0
done=False
while not done:
    action=np.argmax(dqn.dnn.model.predict(np.expand_dims(state, axis=0)),axis=1)[0]
    next_state, reward, done, _ = env.step(action)
    frames.append(env.render(mode = "rgb_array"))
    state=next_state
    num_step+=1
HTML(plot_animation(frames).to_html5_video())