# Landing pad is always at coordinates (0,0). Coordinates are the first
# two numbers in state vector. Reward for moving from the top of the screen
# to landing pad and zero speed is about 100..140 points. If lander moves
# away from landing pad it loses reward back. Episode finishes if the lander
# crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
# Solved is 200 points. Landing outside landing pad is possible. Fuel is
# infinite, so an agent can learn to fly and then land on its first attempt.
# Four discrete actions available: do nothing, fire left orientation engine,
# fire main engine, fire right orientation engine.

from datetime import datetime
import gym
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko

import numpy as np


class DNN:
    def __init__(self):
        self.lr = 0.001

        self.model = km.Sequential()
        self.model.add(kl.Dense(150, input_dim=4, activation="relu"))  # TODO EXO
        self.model.add(kl.Dense(120, activation="relu"))
        self.model.add(kl.Dense(2, activation="linear"))
        self.model.compile(loss='mse', optimizer=ko.Adam(lr=self.lr))  # TODO EXO


class DQN:
    """ Implementation of deep q learning algorithm """

    def __init__(self):

        self.prob_random = 1.0  # Probability to play random action
        self.y = .99  # Discount factor
        self.batch_size = 64  # How many experiences to use for each training step
        self.prob_random_end = .01  # Ending chance of random action
        self.prob_random_decay = .996  # Decrease decay of the prob random
        self.max_episode = 300  # Max number of episodes you are allowes to played to train the game
        self.expected_goal = 200  # Expected goal

        self.dnn = DNN()
        self.env = gym.make('CartPole-v0')

        self.memory = []

        self.metadata = [] # we will store here info score, at the end of each episode


    def save_experience(self, experience):
        self.memory.append(experience)
        self.memory = self.memory[-self.batch_size:]

    def choose_action(self, state, prob_random):
        if np.random.rand() <= prob_random:
            action = np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.dnn.model.predict(state))
        return action

    def run_one_step(self, state):
        action = self.choose_action(state, self.prob_random)
        next_state, reward, done, _ = self.env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        return state, action, reward, next_state, done

    def generate_target_q(self, train_state, train_action, train_reward, train_next_state, train_done):

        # Our predictions (actions to take) from the main Q network
        target_q = self.dnn.model.predict(train_state)

        # Tells us whether game over or not
        # We will multiply our rewards by this value
        # to ensure we don't train on the last move
        train_gameover = train_done == 0

        # Q value of the next state based on action
        target_q_next_state = self.dnn.model.predict(train_next_state)
        train_next_state_values = np.max(target_q_next_state[range(self.batch_size)], axis=1)

        # Reward from the action chosen in the train batch
        actual_reward = train_reward + (self.y * train_next_state_values * train_gameover)
        target_q[range(self.batch_size), train_action] = actual_reward
        return target_q

    def train_one_step(self):

        batch_data = self.memory
        train_state = np.array([i[0] for i in batch_data])
        train_action = np.array([i[1] for i in batch_data])
        train_reward = np.array([i[2] for i in batch_data])
        train_next_state = np.array([i[3] for i in batch_data])
        train_done = np.array([i[4] for i in batch_data])

        # These lines remove useless dimension of the matrix
        train_state = np.squeeze(train_state)
        train_next_state = np.squeeze(train_next_state)

        # Generate target Q
        target_q = self.generate_target_q(
            train_state=train_state,
            train_action=train_action,
            train_reward=train_reward,
            train_next_state=train_next_state,
            train_done=train_done
        )

        loss = self.dnn.model.train_on_batch(train_state, target_q)
        return loss

    def train(self):
        scores = []
        for e in range(self.max_episode):
            # Init New episode
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            episode_score = 0
            while True:
                state, action, reward, next_state, done = self.run_one_step(state)
                self.save_experience(experience=[state, action, reward, next_state, done])
                episode_score += reward
                state = next_state
                if len(self.memory) >= self.batch_size:
                    self.train_one_step()
                    if self.prob_random > self.prob_random_end:
                        self.prob_random *= self.prob_random_decay
                if done:
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    self.metadata.append([now, e, episode_score, self.prob_random])
                    print(
                        "{} - episode: {}/{}, score: {:.1f} - prob_random {:.3f}".format(dt_string, e, self.max_episode,
                                                                                         episode_score,
                                                                                         self.prob_random))
                    break
            scores.append(episode_score)

            # Average score of last 100 episode
            means_last_10_scores = np.mean(scores[-10:])
            if means_last_10_scores == self.expected_goal:
                print('\n Task Completed! \n')
                break
            print("Average over last 10 episode: {0:.2f} \n".format(means_last_10_scores))
        print("Maximum number of episode played: %d" % self.max_episode)

dqn = DQN()
dqn.train()
