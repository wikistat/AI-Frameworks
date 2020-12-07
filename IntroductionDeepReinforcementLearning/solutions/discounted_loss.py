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