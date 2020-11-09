class DNN:
    def __init__(self):

        self.lr = 0.001

        self.model = km.Sequential()
        self.model.add(kl.Dense(150, input_dim=4, activation="relu"))
        self.model.add(kl.Dense(120, activation="relu"))
        self.model.add(kl.Dense(2, activation="linear"))
        self.model.compile(loss='mse', optimizer=ko.Adam(lr=self.lr))