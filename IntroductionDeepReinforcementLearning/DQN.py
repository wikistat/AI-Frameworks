import numpy as np
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
from gridworld import gameEnv

env = gameEnv(partial=False, size=5)


class ExperienceReplay:
    def __init__(self, buffer_size=50000):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,action,reward,next_state,done]
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        """ Adds list of experiences to the buffer """
        # Extend the stored experiences
        self.buffer.extend(experience)
        # Keep the last buffer_size number of experiences
        self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, size):
        """ Returns a sample of experiences from the buffer """
        sample_idxs = np.random.randint(len(self.buffer), size=size)
        sample_output = [self.buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size, -1))
        return sample_output


class Qnetwork():
    def __init__(self):
        self.inputs = kl.Input(shape=[84, 84, 3], name="main_input")

        self.model = kl.Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation="relu",
            padding="valid",
            name="conv1")(self.inputs)
        self.model = kl.Conv2D(
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation="relu",
            padding="valid",
            name="conv2")(self.model)
        self.model = kl.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=[1, 1],
            activation="relu",
            padding="valid",
            name="conv3")(self.model)
        self.model = kl.Conv2D(
            filters=512,
            kernel_size=[7, 7],
            strides=[1, 1],
            activation="relu",
            padding="valid",
            name="conv4")(self.model)

        self.model = kl.Flatten()(self.model)
        self.model = kl.Dense(256, activation="relu")(self.model)
        self.model = kl.Dense(4, activation="linear")(self.model)
        self.model = km.Model(self.inputs, self.model)
        self.model.compile("adam", "mse")
        self.model.optimizer.lr = 0.0001


class DQN:
    def __init__(self):
        self.batch_size = 64  # How many experiences to use for each training step
        self.num_epochs = 20  # How many epochs to train
        self.train_frequency = 5  # How often you update the network
        self.y = 0.99  # Discount factor
        self.prob_random_start = 0.6  # Starting chance of random action
        self.prob_random_end = 0.1  # Ending chance of random action
        self.annealing_steps = 1000.  # Steps of training to reduce from start_e -> end_e
        self.num_episodes = 10000  # How many episodes of game environment to train
        self.min_pre_train_episodes = 100  # Number of episodes of random actions
        self.max_num_step = 50  # Maximum allowed episode length
        self.goal = 15

        # Reset everything from keras session
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
        # Create an experience replay for the current episode.
        experiences_episode = []

        # Get the game state from the environment
        state = env.reset()

        done = False  # Game is complete
        cur_step = 0  # Running sum of number of steps taken in episode

        while cur_step < self.max_num_step and not done:
            cur_step += 1
            if np.random.rand() < prob_random or \
                    num_episode < self.min_pre_train_episodes:
                # Act randomly based on prob_random or if we
                # have not accumulated enough pre_train episodes
                action = np.random.randint(env.actions)
            else:
                # Decide what action to take from the Q network
                action = np.argmax(self.main_qn.model.predict(np.array([state])))

            # Take the action and retrieve the next state, reward and done
            next_state, reward, done = env.step(action)

            # Setup the experience to be stored in the episode buffer
            experience = [state, action, reward, next_state, done]

            # Store the experience in the episode buffer
            experiences_episode.append(experience)

            # Update the state
            state = next_state

        return experiences_episode

    def train_one_step(self):
        # Train batch is [[state,action,reward,next_state,done],...]
        train_batch = self.experience_replay.sample(self.batch_size)

        # Separate the batch into numpy array for each compents
        train_state = np.array([x[0] for x in train_batch])
        train_action = np.array([x[1] for x in train_batch])
        train_reward = np.array([x[2] for x in train_batch])
        train_next_state = np.array([x[3] for x in train_batch])
        train_done = np.array([x[4] for x in train_batch])

        # Our predictions (actions to take) from the main Q network
        target_q = self.main_qn.model.predict(train_state)

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

            experiences_episode = self.run_one_episode(num_episode, prob_random)
            self.experience_replay.add(experiences_episode)

            if num_episode > self.min_pre_train_episodes:
                # Training the network

                if prob_random > self.prob_random_end:
                    # Drop the probability of a random action
                    prob_random -= prob_random_drop

                if num_episode % self.train_frequency == 0:
                    for num_epoch in range(self.num_epochs):
                        loss = self.train_one_step()
                        losses.append(loss)

                    # Update the target model with values from the main model
                    self.update_target_graph()

            # Increment the episode
            num_episode += 1
            num_steps.append(len(experiences_episode))
            rewards.append(sum([e[2] for e in experiences_episode]))

            if num_episode % print_every == 0:
                # Print progress
                mean_loss = np.mean(losses[-(print_every * self.num_epochs):])

                print("Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                    num_episode, np.mean(rewards[-print_every:]), prob_random, mean_loss))
                if np.mean(rewards[-print_every:]) >= self.goal:
                    print("Training complete!")
                    break


##### TEST 1
dqn = DQN()
for target_layer_weight, main_layer_weight in zip(dqn.target_qn.model.get_weights(), dqn.main_qn.model.get_weights()):
    if len(target_layer_weight.shape)>1:
        print(target_layer_weight.shape)
        assert not(np.all(target_layer_weight == main_layer_weight))

dqn.update_target_graph()
for target_layer_weight, main_layer_weight in zip(dqn.target_qn.model.get_weights(), dqn.main_qn.model.get_weights()):
    if len(target_layer_weight.shape)>1:
        print(target_layer_weight.shape)
        assert np.all(target_layer_weight == main_layer_weight)


#### TEST 2


### Final RUn
dqn = DQN()
dqn.train()
