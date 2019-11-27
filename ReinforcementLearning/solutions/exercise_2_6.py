class Qnetwork():
    def __init__(self):
        # The input image of the game is 84 x 84 x 3 (RGB)
        self.inputs = kl.Input(shape=[84, 84, 3], name="main_input")

        # There will be four layers of convolutions performed on the image input
        # A convolution take a portion of an input and matrix multiplies
        # a filter on the portion to get a new input (see below)
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

        # We then separate the final convolution layer into an advantage and value
        # stream. The value function is how well off you are in a given state.
        # The advantage is the how much better off you are after making a particular
        # move. Q is the value function of a state after a given action.
        # Advantage(state, action) = Q(state, action) - Value(state)
        self.stream_AC = kl.Lambda(lambda layer: layer[:, :, :, :256], name="advantage")(self.model)
        self.stream_VC = kl.Lambda(lambda layer: layer[:, :, :, 256:], name="value")(self.model)

        # We then flatten the advantage and value functions
        self.stream_AC = kl.Flatten(name="advantage_flatten")(self.stream_AC)
        self.stream_VC = kl.Flatten(name="value_flatten")(self.stream_VC)

        # We define weights for our advantage and value layers. We will train these
        # layers so the matmul will match the expected value and advantage from play
        self.Advantage = kl.Dense(env.actions, name="advantage_final")(self.stream_AC)
        self.Value = kl.Dense(1, name="value_final")(self.stream_VC)

        # To get the Q output, we need to add the value to the advantage.
        # The advantage to be evaluated will bebased on how good the action
        # is based on the average advantage of that state
        self.model = kl.Lambda(lambda val_adv: val_adv[0] + (val_adv[1] - K.mean(val_adv[1], axis=1, keepdims=True)),
                            name="final_out")([self.Value, self.Advantage])
        self.model = km.Model(self.inputs, self.model)
        self.model.compile("adam", "mse")
        self.model.optimizer.lr = 0.0001
