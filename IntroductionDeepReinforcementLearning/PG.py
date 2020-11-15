import numpy as np
import random
import gym
import math

import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as klo
import tensorflow.keras.backend as K
import tensorflow.keras.metrics as kme


tf.config.experimental_run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')




class discountedLoss(klo.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self,
                 reduction=klo.Reduction.AUTO,
                 name='discountedLoss'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred, adv):
        log_lik = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        loss = K.mean(log_lik * adv, keepdims=True)
        return loss


class kerasModel(km.Model):
    def __init__(self):
        super(kerasModel, self).__init__()
        self.layersList = []
        self.layersList.append(kl.Dense(9, activation="relu",
                     input_shape=(4,),
                     use_bias=False,
                     kernel_initializer=ki.VarianceScaling(),
                     name="dense_1"))
        self.layersList.append(kl.Dense(1,
                       activation="sigmoid",
                       kernel_initializer=ki.VarianceScaling(),
                       use_bias=False,
                       name="out"))

        self.loss = discountedLoss()
        self.optimizer = ko.Adam(lr=1e-2)
        self.train_loss = kme.Mean(name='train_loss')
        self.validation_loss = kme.Mean(name='val_loss')
        self.metric = kme.Accuracy(name="accuracy")

        @tf.function()
        def predict(x):
            """
            This is where we run
            through our whole dataset and return it, when training and testing.
            """
            for l in self.layersList:
                x = l(x)
            return x
        self.predict = predict

        @tf.function()
        def train_step(x, labels, adv):
            """
                This is a TensorFlow function, run once for each epoch for the
                whole input. We move forward first, then calculate gradients with
                Gradient Tape to move backwards.
            """
            with tf.GradientTape() as tape:
                predictions = self.predict(x)
                loss = self.loss.call(
                    y_true=labels,
                    y_pred = predictions,
                    adv = adv)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            return loss

        self.train_step = train_step



def score_model(env, model, num_tests, dimen,):
    scores = []
    for num_test in range(num_tests):
        observation = env.reset()
        reward_sum = 0
        while True:
            state = np.reshape(observation, [1, dimen])
            predict = model.predict(state)[0]
            action = 1 if predict>0.5 else 0
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
    return np.mean(scores)


def discount_rewards(rewards, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
    """
    n_rewards = len(rewards)
    discounted_rewards = [0 for _ in range(n_rewards)]
    for i in range(n_rewards):
        for j,r in enumerate(rewards[i:]):
            discounted_rewards[i]+= math.pow(gamma,j)*r
    return discounted_rewards


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

    def choose_action(self, state):
        predict = self.model.predict(state)[0]
        action = 1 if random.uniform(0, 1) < predict else 0
        return action

    def run_one_step_episode(self, observation):
        # Generate state and action for the current iteration
        state = np.reshape(observation, [1, self.dim_input])
        action = self.choose_action(state)

        # Append the observations and outputs for learning
        self.states = np.vstack([self.states, state])
        self.actions = np.vstack([self.actions, action])

        # Determine the outcome of the action generated
        observation, reward, done, _ = self.env.step(action)

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

        loss = self.model.train_step(
            x = self.states,
            labels = self.actions,
            adv = self.discounted_rewards)
        self.losses.append(float(loss))

        # Clear out game variables
        self.states = np.empty(0).reshape(0, self.dim_input)
        self.actions = np.empty(0).reshape(0, 1)
        self.discounted_rewards = np.empty(0).reshape(0, 1)


    def train(self):
        #########  INIT VALUES  #########
        
        # Init Model to be trained
        self.model = kerasModel()

        # Placeholders for our observations, outputs and rewards
        self.states = np.empty(0).reshape(0, self.dim_input)
        self.actions = np. empty(0).reshape(0, 1)
        self.rewards = np.empty(0).reshape(0, 1)
        self.discounted_rewards = np.empty(0).reshape(0, 1)
        self.losses = []
            
        
        # Number of episode and total score
        num_episode = 0
        reward_sum = 0
        
        # First observation
        observation = self.env.reset()

        while num_episode < self.num_episodes:
            observation, reward, done = self.run_one_step_episode(observation)
            reward_sum += reward

            # If the episode if Over and ifwe have reach 50=batch_size episodes run training for the build batch
            if done:
                if (num_episode + 1) % self.batch_size == 0:
                    self.run_one_batch_train()
                    #print(self.losses)
                    #b
                # Print results periodically
                if (num_episode + 1) % self.print_every == 0:
                    test_score = score_model(self.env, self.model, self.n_test, self.dim_input)
                    print(
                        "Average reward for training episode {}: {:0.2f} Mean test score over {:d} episode: {:0.2f} Loss: {:0.6f} ".format(
                            (num_episode + 1), reward_sum / self.print_every, self.n_test,
                            test_score,
                            self.losses[-1]))
                    reward_sum = 0
                    if test_score >= self.goal:
                        print("Solved in {} episodes!".format(num_episode))
                        break

                num_episode += 1
                observation = self.env.reset()

pg = PG()
pg.train()