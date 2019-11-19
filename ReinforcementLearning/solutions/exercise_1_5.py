class PG:

    def __init__(self, gamma = .99, batch_size = 50, num_episodes = 10000, goal = 190, n_test = 10, print_every = 100):

        self.gamma = gamma      # -> Discounted reward
        self.batch_size = batch_size  # -> Size of episode before training on a batch

        # Stop factor
        self.num_episodes = num_episodes # Max number of iterations
        self.goal = goal           # Stop if our network achieve this goal over *n_test*
        self.n_test = n_test

        self.print_every = print_every #?Numbe rof episode before trying if our model perform well.

    def run_one_step_episode(self, observation):
        # Generate state and action for the current iteration
        state = np.reshape(observation, [1, self.dimen])
        predict = self.model_predict.predict([state])[0]
        action = 0 if random.uniform(0, 1) < predict else 1

        # Append the observations and outputs for learning
        self.states = np.vstack([self.states, state])
        self.actions = np.vstack([self.actions, action])

        # Determine the outcome of the action generated
        observation, reward, done, _ = env.step(action)


        ##Append the rewards for learning
        self.rewards = np.vstack([self.rewards, reward])

        # If the episode if Over
        if done:
            # Computed the discounted rewards for this episode
            discounted_rewards_episode = discount_rewards(self.rewards, self.gamma)

            # Append the discounted rewards for learning
            self.discounted_rewards = np.vstack([self.discounted_rewards, discounted_rewards_episode])
            self.rewards = np.empty(0).reshape(0, 1)

        return observation, reward, done

    def run_one_batch_train(self):
        # Normalize the discounted rewards
        self.discounted_rewards -= self.discounted_rewards.mean()
        self.discounted_rewards /= self.discounted_rewards.std()
        self.discounted_rewards = self.discounted_rewards.squeeze()

        actions_train = 1 - self.actions
        loss = self.model_train.train_on_batch([self.states, self.discounted_rewards], actions_train)
        self.losses.append(loss)

        # Clear out game variables
        self.states = np.empty(0).reshape(0, self.dimen)
        self.actions = np.empty(0).reshape(0, 1)
        self.discounted_rewards = np.empty(0).reshape(0, 1)

    def print_status(self, num_episode, reward_sum, score):
        # Print status
        print(
            "Average reward for training episode {}: {:0.2f} Test Score of {:d} episode: {:0.2f} Loss: {:0.6f} ".format(
                (num_episode + 1), reward_sum / self.print_every, self.n_test,
                score,
                self.losses[-1]))
        return score

    def train(self):

        # Setting up our environment
        observation = env.reset()
        self.dimen = env.reset().shape[0]


        # Placeholders for our observations, outputs and rewards
        self.states = np.empty(0).reshape(0, self.dimen)
        self.actions = np.empty(0).reshape(0, 1)
        self.rewards = np.empty(0).reshape(0, 1)
        self.discounted_rewards = np.empty(0).reshape(0, 1)
        self.losses = []


        num_episode = 0
        reward_sum = 0 # # For display info

        self.model_train, self.model_predict = generate_train_predict_network(hidden_layer_neurons=9, dimen=(self.dimen,))

        while num_episode < self.num_episodes:
            observation, reward, done = self.run_one_step_episode(observation)
            reward_sum += reward

            # If the episode if Over and ifwe have reach 50=batch_size episodes run training for the build batch
            if done:
                if (num_episode + 1) % self.batch_size == 0:
                    self.run_one_batch_train()

                # Print results periodically
                if (num_episode + 1) % self.print_every == 0:
                    score = score_model(self.model_predict, self.n_test, self.dimen)
                    self.print_status(num_episode, reward_sum, score)
                    reward_sum = 0
                    if score >= self.goal:
                        print("Solved in {} episodes!".format(num_episode))
                        break

                num_episode += 1
                observation = env.reset()