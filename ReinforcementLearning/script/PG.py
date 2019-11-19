import numpy as np
import random
import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#Tensorflow/Keras utils
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as klo
import tensorflow.keras.backend as K


# Gym Library
import gym


env = gym.make("CartPole-v0")


def discount_rewards(r, gamma=0.99):
    """Takes 1d float array of rewards and computes discounted reward
    e.g. f([1, 1, 1], 0.99) -> [2.9701, 1.99, 1]
    """
    prior = 0
    out = []
    for val in r:
        new_val = val + prior * gamma
        out.append(new_val)
        prior = new_val
    discounted_rewards = np.array(out[::-1])
    return discounted_rewards



def generate_train_predict_network(hidden_layer_neurons, dimen, lr = 1e-2, initializer = ki.VarianceScaling()):
    num_actions = 1
    inp = kl.Input(shape=dimen,name="input_x")
    adv = kl.Input(shape=[1], name="advantages")
    x = kl.Dense(hidden_layer_neurons,  activation="relu",
                     use_bias=False,
                     kernel_initializer=initializer,
                     name="dense_1")(inp)
    out = kl.Dense(num_actions,
                       activation="sigmoid",
                       kernel_initializer=initializer,
                       use_bias=False,
                       name="out")(x)

    model_train = km.Model(inputs=[inp, adv], outputs=out)
    model_predict = km.Model(inputs=[inp], outputs=out)

    def my_custom_loss(y_true, y_pred):
        log_lik = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(log_lik * adv, keepdims=True)

    model_train.compile(loss=my_custom_loss, optimizer=ko.Adam(lr))
    return model_train, model_predict



# See our trained bot in action
def score_model(model, num_tests, dimen, render=False):
    scores = []
    for num_test in range(num_tests):
        observation = env.reset()
        reward_sum = 0
        while True:
            if render:
                env.render()

            state = np.reshape(observation, [1, dimen])
            predict = model.predict([state])[0]
            action = 0 if predict>0.5 else 1
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            if done:
                break
        scores.append(reward_sum)
    env.close()
    return np.mean(scores)


