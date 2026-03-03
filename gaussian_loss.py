import tensorflow as tf


class gaussian_em_loss:
    """
    Expected complete-data negative log-likelihood under a Gaussian mixture model.

    Derivation
    ----------
    Generative model   :  p(x_{hw} | z_{hw} = k) = N(x_{hw} ; mu_k(h,w) , sigma^2)
    Uniform prior      :  p(z = k) = 1/K

    Complete-data NLL  :  L = -E_gamma[ log p(x, z | mu) ]
                            = sum_k  gamma_k * ||x - mu_k||^2 / (2 * sigma^2)  +  const

    Minimising L over mu_k (M-step) is weighted least squares.
    The sigma acts as a softness parameter for the E-step:
        sigma -> 0  :  hard assignment (degenerates to K-means)
        sigma -> inf:  uniform gamma  (no learning signal)

    Typical choice for [0, 1] normalised images: sigma = 0.25.

    No KL / inter-cluster penalty is needed: the E-step already provides
    well-calibrated soft assignments when sigma is tuned properly.
    """

    def __init__(self, sigma: float = 0.25):
        self.sigma = sigma
        self._two_sigma_sq = tf.constant(2.0 * sigma ** 2, dtype=tf.float32)

    def __call__(
        self,
        predictions: tf.Tensor,  # (B, K, H, W, 1)  predicted means mu_k
        targets: tf.Tensor,      # (B, 1, H, W, 1)  observed images  (broadcast over K)
        gamma: tf.Tensor,        # (B, K, H, W, 1)  responsibilities (stop-gradient)
    ) -> tf.Tensor:
        """
        Weighted MSE loss.

        gamma is detached from the computation graph (E-step is analytical,
        not differentiated through) — consistent with the EM principle where
        the E-step is treated as a fixed inference step.
        """
        sq_err   = tf.square(predictions - targets)       # (B, K, H, W, 1)
        weighted = tf.stop_gradient(gamma) * sq_err       # (B, K, H, W, 1)

        # Sum over components, spatial dims, channels — mean over batch
        loss = tf.reduce_mean(
            tf.reduce_sum(weighted, axis=[1, 2, 3, 4])
        )
        return loss / self._two_sigma_sq
