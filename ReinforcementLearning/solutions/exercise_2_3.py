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
        self.model = kl.Dense(4, activation = "linear")(self.model)
        self.model = km.Model(self.inputs, self.model)
        self.model.compile("adam", "mse")
        self.model.optimizer.lr = 0.0001