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
import pickle
import numpy as np



class DNN:
    def __init__(self):

        self.lr = 0.001

        self.model = km.Sequential()
        self.model.add(kl.Dense(150, input_dim=8, activation="relu")) # TODO EXO
        self.model.add(kl.Dense(120, activation="relu"))
        self.model.add(kl.Dense(4, activation="linear"))
        self.model.compile(loss='mse', optimizer=ko.Adam(lr=self.lr)) # TODO EXO



class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self):

        self.prob_random = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.prob_random_min = .01
        self.prob_random_decay = .996
        self.max_episode = 400
        self.max_steps = 3000 # TODO Useless because its 1000 by default
        
        self.dnn = DNN()
        self.env = gym.make('LunarLander-v2')

        self.memory = []

    def save_experience(self,experience):
        self.memory.append(experience)
        if len(self.memory)>self.batch_size+1:
            self.memory=self.memory[1:]


    def choose_action(self, state):
        if np.random.rand() <= self.prob_random:
            action =  np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.dnn.model.predict(state))
        return action

    def run_one_step(self, state, score):
        action = self.choose_action(state)
        # env.render()
        next_state, reward, done, _ = self.env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        self.save_experience(experience = [state, action, reward, next_state, done])

        score += reward
        state = next_state
        return state, score, done

    def train_one_step(self):


        batch_data = self.memory[:self.batch_size]
        states = np.array([i[0] for i in batch_data])
        actions = np.array([i[1] for i in batch_data])
        rewards = np.array([i[2] for i in batch_data])
        next_states = np.array([i[3] for i in batch_data])
        dones = np.array([i[4] for i in batch_data])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.dnn.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.dnn.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        loss = self.dnn.model.train_on_batch(states, targets_full)
        if self.prob_random > self.prob_random_min:
            self.prob_random *= self.prob_random_decay
        return loss

    def train(self):
        metadata = []
        scores = []
        loss=-1
        for e in range(self.max_episode):
            # Init New episode
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            score = 0
            for i in range(self.max_steps):
                state, score, done = self.run_one_step(state, score)

                if len(self.memory)>self.batch_size:
                    loss = self.train_one_step()
                if done:
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    metadata.append([now, e, i, score, loss, self.prob_random])
                    pickle.dump(metadata, open("lunar_lander_metadata.pkl", "wb"))
                    print("{} - episode: {}/{}, n_step{}, score: {:.1f} - loss {} - prob_random {:.3f}".format(dt_string,e, self.max_episode, i, score, loss, self.prob_random))
                    break
            scores.append(score)

            # Average score of last 100 episode
            is_solved = np.mean(scores[-100:])
            if is_solved > 200:
                print('\n Task Completed! \n')
                break
            print("Average over last 100 episode: {0:.2f} \n".format(is_solved))





dqn = DQN()
dqn.train()
