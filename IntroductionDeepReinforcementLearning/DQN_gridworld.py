from datetime import datetime
import numpy as np
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
from gridworld import gameEnv
from experience_replay import ExperienceReplay



class Qnetwork:
    def __init__(self):
        self.inputs = kl.Input(shape=??, name="main_input")

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
        self.model = kl.Dense(??, activation=??)(self.model)
        self.model = km.Model(self.inputs, self.model)
        self.model.compile("adam", "mse")
        self.model.optimizer.lr = 0.0001


class DQN:
    def __init__(self):
        self.batch_size = 64  # How many experiences to use for each training step
        self.train_frequency = 5  # How often you update the network
        self.num_epochs = 20  # How many epochs to train when updating the network
        self.y = 0.99  # Discount factor
        self.prob_random_start = 0.6  # Starting chance of random action
        self.prob_random_end = 0.1  # Ending chance of random action
        self.annealing_steps = 1000.  # Steps of training to reduce from start_e -> end_e
        self.max_num_episodes = 10000  # Max number of episodes you are allowes to played to train the game
        self.min_pre_train_episodes = 100  # Number of episodes played with random actions before to start training.
        self.max_num_step = 50  # Maximum allowed episode length
        self.goal = 15 # Number of rewards we want to achieve while playing a game.

        # Set env
        self.env = gameEnv(partial=False, size=5)

        # Reset everything from keras session
        K.clear_session()

        # Setup our Q-networks
        self.main_qn = Qnetwork()
        self.target_qn = Qnetwork()

        # Setup our experience replay
        self.experience_replay = ExperienceReplay()

    def update_target_graph(self):
        # TODO
        return

    def choose_action(self, state, prob_random, num_episode):
        # TODO
        return action

    def run_one_episode(self, num_episode, prob_random):
        # TODO
        return experiences_episode

    def generate_target_q(self, train_state, train_action, train_reward, train_next_state, train_done):
        # TODO
        return target_q

    def train_one_step(self):
        # Train batch is [[state,action,reward,next_state,done],...]
        train_batch = self.experience_replay.sample(self.batch_size)

        # Separate the batch into numpy array for each compents
        train_state = np.array([x[0] for x in train_batch])
        train_action = np.array([x[1] for x in train_batch])
        train_reward = np.array([x[2] for x in train_batch])
        train_next_state = np.array([x[3] for x in train_batch])
        train_done = np.array([x[4] for x in train_batch])

        # Generate target Q
        target_q = self.generate_target_q(
            train_state=train_state,
            train_action=train_action,
            train_reward=train_reward,
            train_next_state=train_next_state,
            train_done=train_done
        )

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

        # Init variable
        num_steps = []  # Tracks number of steps per episode
        rewards = []  # Tracks rewards per episode
        print_every = 50  # How often to print status
        losses = [0]  # Tracking training losses
        num_episode = 0

        while True:
            # Run one episode
            experiences_episode = self.run_one_episode(num_episode, prob_random)

            # Save the episode in the replay buffer
            self.experience_replay.add(experiences_episode)

            # If we have play enoug episode. Start the training
            if num_episode > self.min_pre_train_episodes:

                # Drop the probability of a random action if wi didn't reach the prob_random_end value
                if prob_random > self.prob_random_end:
                    prob_random -= prob_random_drop

                # Every train_frequency iteration, train the model
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

            # Print Info
            if num_episode % print_every == 0:
                # datetime object containing current date and time
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                mean_loss = np.mean(losses[-(print_every * self.num_epochs):])
                print("{} - Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                    dt_string, num_episode, np.mean(rewards[-print_every:]), prob_random, mean_loss))

            # Stop Condition
            if np.mean(rewards[-print_every:]) >= self.goal:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                mean_loss = np.mean(losses[-(print_every * self.num_epochs):])
                print("{} - Num episode: {} Mean reward: {:0.4f} Prob random: {:0.4f}, Loss: {:0.04f}".format(
                    dt_string, num_episode, np.mean(rewards[-print_every:]), prob_random, mean_loss))
                print("Training complete because we reached goal rewards.")
                break
            if num_episode > self.max_num_episodes:
                print("Training Stop because we reached max num of episodes")
                break


### Final RUn
dqn = DQN()
#dqn.train()
