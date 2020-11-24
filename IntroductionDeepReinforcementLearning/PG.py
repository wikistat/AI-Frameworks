import numpy as np
import random
import gym

import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')
from keras_model import kerasModel
from discounted_rewards import discount_rewards


class PG:

    def __init__(self, gamma=.99, batch_size=50, num_episodes=10000, goal=190, n_test=10, print_every=100):
        # Environment
        self.env = gym.make("CartPole-v0")
        self.dim_input = self.env.observation_space.shape[0]

        # Parameters
        self.gamma = gamma  # -> Discounted reward
        self.batch_size = batch_size  # -> Size of episode before training on a batch

        # Stop factor
        self.num_episodes = num_episodes  # Max number of iterations
        self.goal = goal  # Stop if our network achieve this goal over *n_test*
        self.n_test = n_test
        self.print_every = print_every  # ?Numbe rof episode before trying if our model perform well.

        # Init Model to be trained
        self.model = kerasModel()

        # Placeholders for our observations, outputs and rewards
        self.experiences = []
        self.losses = []

    def choose_action(self, state):
        #TODO
        return action

    def run_one_episode(self):
        #TODO
        return score

    def run_one_batch_train(self):
        # TODO
        return loss

    def score_model(self, model, num_tests, dimen, ):
        scores = []
        for num_test in range(num_tests):
            observation = self.env.reset()
            reward_sum = 0
            while True:
                state = np.reshape(observation, [1, dimen])
                predict = model.predict(state)[0]
                action = 1 if predict > 0.5 else 0
                observation, reward, done, _ = self.env.step(action)
                reward_sum += reward
                if done:
                    break
            scores.append(reward_sum)
        return np.mean(scores)

    def train(self):
        metadata = []
        i_batch = 0
        # Number of episode and total score
        num_episode = 0
        train_score_sum = 0

        while num_episode < self.num_episodes:
            train_score = self.run_one_episode()
            train_score_sum += train_score
            num_episode += 1

            if num_episode % self.batch_size == 0:
                i_batch += 1
                loss = self.run_one_batch_train()
                self.losses.append(loss)
                metadata.append([i_batch, self.score_model(self.model, self.n_test, self.dim_input)])

            # Print results periodically
            if num_episode % self.print_every == 0:
                test_score = self.score_model(self.model, self.n_test, self.dim_input)
                print(
                    "Average reward for training episode {}: {:0.2f} Mean test score over {:d} episode: {:0.2f} Loss: {:0.6f} ".format(
                        num_episode, train_score_sum / self.print_every, self.n_test,
                        test_score,
                        self.losses[-1]))
                reward_sum = 0
                if test_score >= self.goal:
                    print("Solved in {} episodes!".format(num_episode))
                    break
        return metadata


pg = PG()
pg.train()
