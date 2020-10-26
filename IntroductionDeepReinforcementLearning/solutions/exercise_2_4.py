class DQN:
    def __init__(self):
        self.batch_size = 64  # How many experiences to use for each training step
        self.num_epochs = 20  # How many epochs to train
        self.update_freq = 5  # How often you update the network
        self.y = 0.99  # Discount factor
        self.prob_random_start = 0.6  # Starting chance of random action
        self.prob_random_end = 0.1  # Ending chance of random action
        self.annealing_steps = 1000.  # Steps of training to reduce from start_e -> end_e
        self.num_episodes = 10000  # How many episodes of game environment to train
        self.pre_train_episodes = 100  # Number of episodes of random actions
        self.max_num_step = 50  # Maximum allowed episode length
        self. goal = 15

        # Reset everything
        K.clear_session()

        # Setup our Q-networks
        self.main_qn = Qnetwork()
        self.target_qn = Qnetwork()

        # Setup our experience replay
        self.experience_replay = ExperienceReplay()

    def update_target_graph(self):
        updated_weights = np.array(self.main_qn.model.get_weights())
        self.target_qn.model.set_weights(updated_weights)


    def run_one_episode(self, num_episode, prob_random):
        # Create an experience replay for the current episode
        episode_buffer = ExperienceReplay()

        # Get the game state from the environment
        state = env.reset()

        done = False  # Game is complete
        sum_rewards = 0  # Running sum of rewards in episode
        cur_step = 0  # Running sum of number of steps taken in episode

        while cur_step < self.max_num_step and not done:
            cur_step += 1
            if np.random.rand() < prob_random or \
                    num_episode < self.pre_train_episodes:
                # Act randomly based on prob_random or if we
                # have not accumulated enough pre_train episodes
                action = np.random.randint(env.actions)
            else:
                # Decide what action to take from the Q network
                action = np.argmax(self.main_qn.model.predict(np.array([state])))

            # Take the action and retrieve the next state, reward and done
            next_state, reward, done = env.step(action)

            # Setup the episode to be stored in the episode buffer
            episode = np.array([[state], action, reward, [next_state], done])
            episode = episode.reshape(1, -1)

            # Store the experience in the episode buffer
            episode_buffer.add(episode)

            # Update the running rewards
            sum_rewards += reward

            # Update the state
            state = next_state

        return episode_buffer, sum_rewards, cur_step

    def train_one_step(self):
        # Train batch is [[state,action,reward,next_state,done],...]
        train_batch = self.experience_replay.sample(self.batch_size)

        # Separate the batch into its components
        train_state, train_action, train_reward, \
        train_next_state, train_done = train_batch.T

        # Convert the action array into an array of ints so they can be used for indexing
        train_action = train_action.astype(np.int)

        # Stack the train_state and train_next_state for learning
        train_state = np.vstack(train_state)
        train_next_state = np.vstack(train_next_state)

        # Our predictions (actions to take) from the main Q network
        target_q = self.target_qn.model.predict(train_state)

        # Tells us whether game over or not
        # We will multiply our rewards by this value
        # to ensure we don't train on the last move
        train_gameover = train_done == 0

        # Q value of the next state based on action
        target_q_next_state = self.target_qn.model.predict(train_next_state)
        train_next_state_values = np.max(target_q_next_state[range(self.batch_size)], axis=1)

        # Reward from the action chosen in the train batch
        actual_reward = train_reward + (self.y * train_next_state_values * train_gameover)
        target_q[range(self.batch_size), train_action] = actual_reward

        # Train the main model
        loss = self.main_qn.model.train_on_batch(train_state, target_q)
        return loss


    def train(self):

        # Make the networks equal
        self.update_target_graph()

        # We'll begin by acting complete randomly. As we gain experience and improve,
        # we will begin reducing the probability of acting randomly, and instead
        # take the actions that our Q network suggests
        prob_random = self.prob_random_start
        prob_random_drop = (self.prob_random_start - self.prob_random_end) / self.annealing_steps

        num_steps = []  # Tracks number of steps per episode
        rewards = []  # Tracks rewards per episode
        print_every = 50  # How often to print status
        losses = [0]  # Tracking training losses
        num_episode = 0

        while num_episode < self.num_episodes:

            episode_buffer, sum_rewards, cur_step = self.run_one_episode(num_episode, prob_random)

            if num_episode > self.pre_train_episodes:
                # Training the network

                if prob_random > self.prob_random_end:
                    # Drop the probability of a random action
                    prob_random -= prob_random_drop

                if num_episode % self.update_freq == 0:
                    for num_epoch in range(self.num_epochs):
                        loss = self.train_one_step()
                        losses.append(loss)

                    # Update the target model with values from the main model
                    self.update_target_graph()

            # Increment the episode
            num_episode += 1

            self.experience_replay.add(episode_buffer.buffer)
            num_steps.append(cur_step)
            rewards.append(sum_rewards)

            if num_episode % print_every == 0:
                # Print progress
                mean_loss = np.mean(losses[-(print_every * self.num_epochs):])

                print("Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                    num_episode, np.mean(rewards[-print_every:]), prob_random, mean_loss))
                if np.mean(rewards[-print_every:]) >= self.goal:
                    print("Training complete!")
                    break

