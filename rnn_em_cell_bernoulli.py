import tensorflow as tf
import numpy as np 

class rnn_em(object):

    def __init__(self, q_graph, input_shape):
        self.model = q_graph
        # (H, W, 1)
        self.input_shape = input_shape
        

    def initial_state(self, batch_size, K=3):
        """
        Initialize the RNN state with better strategies for initialization
        Uses a more diverse initialization for predictions, which helps EM converge better
        """
        # initial rnn hidden state
        rnn_state = None

        # initial prediction,  shape : (batch_size, K, W, H, 1)
        pred_shape = tf.stack([batch_size, K] + list(self.input_shape))
        
        # Initialize clusters with more diverse values to help EM separate them better
        # Strategy: Initialize clusters at different intensity ranges
        # Cluster 1: low intensity (0.1-0.3)
        # Cluster 2: medium intensity (0.4-0.6)
        # Cluster 3: high intensity (0.7-0.9)
        # This mimics K-means like initialization with different centroids
        
        # Create a base random initialization
        base_init = tf.random.uniform(shape=pred_shape, minval=0.1, maxval=0.9, dtype=tf.float32)
        
        # Apply different intensity ranges for different clusters
        intensity_ranges = []
        range_width = 0.8 / K  # Divide 0.1-0.9 into K ranges
        
        for k in range(K):
            min_val = 0.1 + k * range_width
            max_val = 0.1 + (k + 1) * range_width
            
            # Create a mask for this cluster
            cluster_mask = tf.zeros(pred_shape, dtype=tf.float32)
            cluster_mask = tf.tensor_scatter_nd_update(
                cluster_mask,
                tf.constant([[i, k, 0, 0, 0] for i in range(batch_size)]),
                tf.ones([batch_size], dtype=tf.float32)
            )
            
            # Scale values to desired range
            scaled_values = (base_init * range_width) + min_val
            
            # Apply mask
            intensity_ranges.append(cluster_mask * scaled_values)
        
        # Combine all clusters
        pred = tf.add_n(intensity_ranges)
        
        # Add some noise to avoid symmetric solutions
        noise = tf.random.normal(shape=pred_shape, mean=0.0, stddev=0.05, dtype=tf.float32)
        pred = tf.clip_by_value(pred + noise, 0.01, 0.99)  # Keep in valid probability range

        # initial gamma with better initialization
        # Initialize with a soft clustering assignment rather than completely random
        shape_gamma = tf.stack([batch_size, K] + list(self.input_shape))
        
        # Begin with more balanced responsibilities
        base_gamma = tf.ones(shape=shape_gamma, dtype=tf.float32) / K
        
        # Add small random variations to break symmetry
        gamma_noise = tf.random.normal(shape=shape_gamma, mean=0.0, stddev=0.1, dtype=tf.float32)
        gamma = tf.abs(base_gamma + gamma_noise)
        
        # Ensure gamma sums to 1 along the cluster dimension
        gamma /= tf.reduce_sum(gamma, axis=1, keepdims=True)
        
        return rnn_state, pred, gamma
    
    @staticmethod
    def _q_graph_input(inputs, gamma):
        # this implments gamma * (inputs) without backprop through the gamma variable
        # gamma is processed through the e-step 
        return inputs * tf.stop_gradient(gamma)

    def q_graph_call(self, q_input, rnn_state):
        
        q_shape = tf.shape(q_input)
        M = tf.math.reduce_prod(list(self.input_shape))
        reshaped_q_input = tf.reshape(q_input, 
                                      shape=tf.stack([q_shape[0] * q_shape[1], M])
                                      )
        predictions, rnn_state = self.model(reshaped_q_input, rnn_state)
        return tf.reshape(predictions, shape=q_shape), rnn_state 
    
    def _compute_joint_probs(self, predictions, data):
        """
        computing the joint p(data, z| mu, sigma)
        with sigma fixed (see the article)
        Using log-space for better numerical stability
        """
        # Calculate log probabilities in log space for numerical stability
        log_p_zx = data * tf.math.log(tf.clip_by_value(predictions, 1e-6, 1.0)) + \
                  (1 - data) * tf.math.log(tf.clip_by_value(1 - predictions, 1e-6, 1.0))
        
        # Sum over the feature dimensions
        log_p_zx = tf.reduce_sum(log_p_zx, axis=4, keepdims=True)
        
        # Convert back to probability space
        p_zx = tf.math.exp(log_p_zx)
        
        return p_zx
    def _e_step(self, predictions , targets):
        """
        Performing Expectation step of the EM algorithm: gamma  = P(z |targets, predictions)
        """

        probs = self._compute_joint_probs(predictions, targets)

        # summing up over all z's scenarios 
        # print("probs", tf.shape(probs))
        normalization_const = tf.reduce_sum(probs, axis=1, keepdims=True)
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
        # print("gamma:", gamma.get_shape())

        return gamma
    
    def __call__(self, inputs, state):

        features, targets = inputs
        rnn_state, preds, gamma = state
        delta = preds - features
        q_inputs = self._q_graph_input(delta, gamma)
        q_output, rnn_state = self.q_graph_call(q_inputs, rnn_state)
        gamma = self._e_step(q_output, targets)

        return (rnn_state, q_output, gamma)

        


