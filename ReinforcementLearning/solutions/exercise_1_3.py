frames = []
env.reset()    
observation, reward, done, _ = env.step(env.action_space.sample())
reward_sum = 0        
while not(done):
    img = env.render(mode = "rgb_array")
    env.close()
    frames.append(img)
    reward_sum += reward
    position, velocity, angle, angular_velocity = observation
    if angle < 0:
        action = 0
    else:
        action = 1
    observation, reward, done, _ = env.step(action)
HTML(plot_animation(frames).to_html5_video())