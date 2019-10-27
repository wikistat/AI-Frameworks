observation = env.reset()
reward_sum = 0
all_reward_sum = []
num_games = 100
num_game = 0
while num_game < num_games:
    
    position, velocity, angle, angular_velocity = observation
    if angle < 0:
        action = 0
    else:
        action = 1
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        if num_game %10 == 0:
            print("Game played : %d. Reward for the last 10 episode: %s" %(num_game,all_reward_sum[-10:]) )
        all_reward_sum.append(reward_sum)
        reward_sum = 0
        num_game += 1
        env.reset()
        
print("Over %d episodes, mean reward: %d, std : %d" %(num_games, np.mean(all_reward_sum), np.std(all_reward_sum)))