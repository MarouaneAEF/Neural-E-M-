import tensorflow as tf
import numpy as np 

class rnn_em(object):

    def __init__(self, q_graph, input_shape):
        self.model = q_graph
        # (H, W, 1)
        self.input_shape = input_shape
        

    def initial_state(self, batch_size, K):
        # initial rnn hidden state
        rnn_state = None

        # initial prediction,  shape : (batch_size, K, W, H, 1)
        pred_shape = tf.stack([batch_size, K] + list(self.input_shape))
        
        pred  = .5 * tf.ones(shape=pred_shape, dtype=tf.float32)
        # initial gamma, shape (batch_size, K, W, H, 1)
        shape_gamma = tf.stack([batch_size, K] + list(self.input_shape))
        gamma = tf.abs(tf.random.uniform(shape=shape_gamma, dtype=tf.float32, maxval=1))
        # p(z) prior 
        gamma /= tf.reduce_sum(gamma, axis=1, keepdims=True)  
        
        return rnn_state, pred, gamma
    
    @staticmethod
    def _q_graph_input(inputs, gamma):
        # this implments gamma * (inputs) without backprop through the gamma variable
        # gamma is processed through the e-step 
        return inputs * tf.stop_gradient(gamma)

    def q_graph_call(self, q_input, rnn_state):
        
        q_shape = tf.shape(q_input)
        faltten_img_shape = tf.math.reduce_prod(list(self.input_shape))
        # reshaped_q_input = tf.reshape(q_input, 
        #                               shape=tf.stack([q_shape[0] * q_shape[1], faltten_img_shape])
        #                               )
        predictions, rnn_state = self.model(q_input, rnn_state)

        return tf.reshape(predictions, shape=q_shape), rnn_state 
    
    def _compute_joint_probs(self, predictions, data):
        """
        computing the joint p(data, z| mu, sigma)
        with sigma fixed (see the article)
        """
        # print(f"log(predictions): {tf.math.log(predictions)}")
        # non normalized joint probability p(z,x):
        p_zx = (tf.reduce_sum(
            data * tf.math.log(predictions) + (1 - data) * tf.math.log(1 - (predictions))
            , axis=-1))
        # print(f"p-zx: {p_zx}")
        # for each value data_x in data and the corresponding value mu_x in mu, 
        # this line computes the joint probability p(data, z| mu, sigma): 
        probs = tf.exp(p_zx) 
        return probs 
    def _e_step(self, predictions , targets):
        """
        Performing Expectation step of the EM algorithm: gamma  = P(z |targets, predictions)
        """

        probs = self._compute_joint_probs(predictions, targets)

        # print(f"probs: {probs}")
        
        # summing up over all z's scenarios 
        # print("probs", tf.shape(probs))
        normalization_const = tf.reduce_sum(tf.exp(probs), axis=1, keepdims=True)
        # print(f"normalization_const: {normalization_const}")
        # p(z|x,psi)
        gamma = probs / normalization_const
        # print(f"gamma:{gamma}")
        # gamma represents the responsibility of each mixture component for generating each observation in the input data. 
        # It is a tensor with the same shape as probs, which has dimensions (B, K, W, H, 1), 
        # where B is the batch size, K is the number of mixture components, and W, H, and 1 represent the dimensions of 
        # the input data. Each element in gamma is the probability of that mixture component generating the corresponding 
        # observation in the input data. 
        # For example, gamma[i, j, k, l, m] represents the probability that the jth mixture component generated 
        # the mth channel of the pixel at position (k, l) in the ith image in the batch. 
        # The sum of gamma over the second dimension gives a tensor with dimensions (B, W, H, 1) that represents 
        # the valid total probability of each observation in the input data being generated by any of the mixture components.
        

        return gamma
    
    def __call__(self, inputs, state):

        features, targets = inputs
        rnn_state, preds, gamma = state
        print(f"preds: {preds.get_shape()}")
        delta = preds - features
        q_inputs = self._q_graph_input(delta, gamma)
        q_output, rnn_state = self.q_graph_call(q_inputs, rnn_state)
        gamma = self._e_step(q_output, targets)

        return (rnn_state, q_output, gamma)

        


