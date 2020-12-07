import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as klo
import tensorflow.keras.backend as K
import tensorflow.keras.metrics as kme

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

    def call(self, y_true, y_pred, disc_r):
        log_lik = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        loss = K.mean(log_lik * disc_r, keepdims=True)
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
        def train_step(x, labels, disc_r):
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
                    disc_r = disc_r)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            self.train_loss(loss)
            return loss

        self.train_step = train_step