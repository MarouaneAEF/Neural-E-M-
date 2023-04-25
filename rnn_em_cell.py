import tensorflow as tf
import tensorflow_probability as tfp 
import numpy as np 

class rnn_em:

    def __init__(self, q_graph, input_shape):
        self.model = q_graph
        # (K, W, H, C)
        self.input_shape = input_shape
        # (K, W, H, 1)
        self.gamma_shape = list(self.input_shape)[:-1] + [1]

    def initial_state(self, batch_size, K):
        # initial rnn hidden state
        h = None
        # initial prediction,  shape : (batch_size, K, W, H, C)
        pred_shape = tf.stack([batch_size, K] + list(self.input_shape))
        pred  = tf.zeros(shape=pred_shape, dtype=tf.float32)
        # initial gamma, shape (batch_size, K, W, H, 1)
        shape_gama = tf.stack([batch_size, K] + self.gamma_shape)
        # for each pixel i gamma_i = p(z|x) z's: represent targets 
        gamma = tf.abs(tf.random.normal(shape=shape_gama, dtype=tf.float32))
        # q(z|x)
        gamma /= tf.reduce_sum(gamma, axis=1)  

        return h, pred, gamma
    
    @staticmethod
    def q_graph_input(inputs, gamma):
        # this implments gamma * (pred - data) without backprop through the gamma variable
        # gamma is processed through the e-step 
        return inputs * tf.stop_gradient(gamma)

    @staticmethod
    def q_graph_call(self, q_input, h):
        
        q_shape = tf.shape(q_input)
        M = tf.math.reduce_prod(list(self.input_shape))
        reshaped_q_input = tf.reshape(q_input, 
                                      shape=tf.stack([q_shape[0] * q_shape[1], M])
                                      )
        predictions, h = self.model(reshaped_q_input, h)

        return tf.reshape(predictions, shape=q_shape), h 
    
    def compute_probs(self, predictions, data):

        mu , sigma = predictions, .25 
        probs = (
                (1 / tf.sqrt((2 * np.pi * sigma ** 2))) * 
                     tf.exp(-(data - mu) ** 2 / (2 * sigma ** 2))
                 )
        # p(x|z)
        probs = tf.reduce_sum(probs, axis=-1, keepdims=True) + 1e-6

    def e_step(self, predictions , targets):

        probs = self.compute_probs(predictions, targets)
        # marginalize over all z's scenarios
        normalization_const = tf.reduce_sum(probs, 1, keepdims=True)
        gamma = probs / normalization_const
        return gamma
    
    def __call__(self, inputs, init_state):

        features, targets = inputs
        h, preds, gamma = init_state
        delta = preds - features
        q_inputs = self.q_graph_input(delta, gamma)
        q_output, h = self.q_graph_call(q_inputs, h)
        gamma = self.e_step(q_output, targets)

        return h, preds, gamma

        


