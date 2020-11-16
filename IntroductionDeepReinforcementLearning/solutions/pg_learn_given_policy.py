class PG:

    def __init__(self):
        # Environment
        self.env = gym.make("CartPole-v0")
        self.dim_input = self.env.observation_space.shape[0]

        # Model
        self.model = self.init_model()
        self.n_episode_max = 1000

    def init_model(self):

        # Build the neural network
        policy_network = km.Sequential()
        policy_network.add(kl.Dense(9, input_shape=(self.dim_input,), activation="relu"))
        policy_network.add(kl.Dense(1, activation="sigmoid"))
        policy_network.compile(loss='binary_crossentropy', optimizer=ko.Adam(), metrics=['accuracy'])
        return policy_network

    def play_one_episode(self):
        train_data = []
        observation = self.env.reset()
        action = 0 if observation[2] < 0 else 1
        done = False
        while not done:
            observation, reward, done, _ = self.env.step(action)
            action = 0 if observation[2] < 0 else 1
            train_data.append([observation, action])
        return train_data

    def train(self):

        for iteration in tqdm(range(self.n_episode_max)):
            train_data = self.play_one_episode()
            n_step = len(train_data)
            target = np.array([x[1] for x in train_data]).reshape((n_step, 1))
            observations = np.array([x[0] for x in train_data])
            self.model.train_on_batch(observations, target)