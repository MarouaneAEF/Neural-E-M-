import tensorflow as tf


class rnn_em_gaussian:
    """
    Neural Expectation-Maximization cell — Gaussian observation model.

    Observation model
    -----------------
        p(x_{hw} | z_{hw} = k) = N(x_{hw} ; mu_k(h,w) , sigma^2)

    E-step  (analytical, per spatial location)
    -------------------------------------------
        log gamma_k(h,w)  ∝  -||x_{hw} - mu_k(h,w)||^2 / (2 * sigma^2)
        gamma_k           =  softmax_k( log gamma_k )

    Computed in log-space (subtract max before exp) for numerical stability
    when sigma is small and squared errors are large.

    M-step  (neural)
    ----------------
    The Q-graph receives  gamma_k * (mu_k - x)  as input — the
    responsibility-weighted prediction error, which is proportional to
    the gradient of the loss w.r.t. mu_k:

        dL / d(mu_k) = gamma_k * (mu_k - x) / sigma^2

    The LSTM can thus learn to implement an adaptive gradient descent
    procedure over EM iterations, integrating curvature information.

    Parameters
    ----------
    q_graph      : Q_graph instance (LSTM encoder-decoder)
    input_shape  : (H, W, C) spatial shape of one image
    sigma        : fixed std-dev of all Gaussian components
    """

    def __init__(self, q_graph, input_shape, sigma: float = 0.25):
        self.model       = q_graph
        self.input_shape = input_shape
        self.sigma       = sigma
        self._two_sigma_sq = tf.constant(2.0 * sigma ** 2, dtype=tf.float32)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int, K: int = 3):
        """
        Initial hidden state for a new sequence / batch.

        mu_k  : initialised at K evenly-spaced intensity levels
                (0.25, 0.50, 0.75 for K=3) + small Gaussian noise.
                This gives the E-step a head-start by breaking symmetry
                in a structured way — each component starts near a
                different intensity quintile.

        gamma : uniform (1/K) + small noise, re-normalised.
        """
        rnn_state = None
        shape     = tf.stack([batch_size, K] + list(self.input_shape))

        # K evenly-spaced intensity priors
        levels = [(k + 1) / (K + 1) for k in range(K)]   # e.g. [0.25, 0.50, 0.75]
        pred_slices = []
        for level in levels:
            noise = tf.random.normal(
                tf.stack([batch_size, 1] + list(self.input_shape)), stddev=0.05)
            prior = tf.fill(
                tf.stack([batch_size, 1] + list(self.input_shape)), float(level))
            pred_slices.append(tf.clip_by_value(prior + noise, 0.01, 0.99))
        pred = tf.concat(pred_slices, axis=1)              # (B, K, H, W, 1)

        # Uniform gamma with symmetry-breaking noise
        gamma = tf.abs(
            tf.ones(shape, dtype=tf.float32) / K
            + tf.random.normal(shape, mean=0.0, stddev=0.05)
        )
        gamma = gamma / tf.reduce_sum(gamma, axis=1, keepdims=True)

        return rnn_state, pred, gamma

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _q_graph_input(delta, gamma):
        """Responsibility-weighted prediction error: gamma_k * (mu_k - x).
        Stop-gradient on gamma: the E-step is not differentiated through."""
        return delta * tf.stop_gradient(gamma)

    def q_graph_call(self, q_input, rnn_state, training: bool = False):
        q_shape  = tf.shape(q_input)
        M        = tf.math.reduce_prod(list(self.input_shape))
        flat     = tf.reshape(
            q_input, tf.stack([q_shape[0] * q_shape[1], M]))
        preds, rnn_state = self.model(flat, rnn_state, training=training)
        return tf.reshape(preds, shape=q_shape), rnn_state

    # ------------------------------------------------------------------
    # E-step  (Gaussian, log-space softmax)
    # ------------------------------------------------------------------

    def _e_step(self, predictions: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """
        Compute pixel-wise responsibilities under the Gaussian model.

        gamma_k(h,w) = softmax_k( -||x_{hw} - mu_k(h,w)||^2 / (2 sigma^2) )

        Implementation note: we work in log-space and subtract the
        per-pixel maximum before exponentiation to avoid underflow when
        sigma is small (tight clusters).

        Shapes
        ------
        predictions : (B, K, H, W, 1)
        targets     : (B, 1, H, W, 1)  —  broadcast over K
        returns     : (B, K, H, W, 1)
        """
        # Unnormalized log-responsibility: -||x - mu_k||^2 / (2 sigma^2)
        log_resp = -tf.square(targets - predictions)           # (B, K, H, W, 1)
        log_resp = tf.reduce_sum(log_resp, axis=4, keepdims=True)  # sum over C
        log_resp = log_resp / self._two_sigma_sq

        # Stabilize via log-sum-exp trick (subtract max over K)
        log_resp -= tf.reduce_max(log_resp, axis=1, keepdims=True)
        resp      = tf.exp(log_resp)
        gamma     = resp / (tf.reduce_sum(resp, axis=1, keepdims=True) + 1e-10)
        return gamma

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, inputs, state):
        features, targets = inputs
        rnn_state, preds, gamma = state

        delta    = preds - features                         # (B, K, H, W, 1)
        q_inputs = self._q_graph_input(delta, gamma)
        q_output, rnn_state = self.q_graph_call(q_inputs, rnn_state)
        gamma    = self._e_step(q_output, targets)

        return (rnn_state, q_output, gamma)
