import tensorflow as tf 

class em_loss(object):

    def __init__(self, prior=0., initial_kl_weight=0.01, max_kl_weight=0.5, annealing_rate=0.001):
        self.prior = prior
        self.initial_kl_weight = initial_kl_weight  # Starting weight for KL term
        self.max_kl_weight = max_kl_weight  # Maximum weight for KL term
        self.annealing_rate = annealing_rate  # Rate of increase per step
        self.current_kl_weight = tf.Variable(initial_kl_weight, trainable=False)
        self.step = tf.Variable(0, trainable=False)
        

    @staticmethod
    def cross_entropy_loss(samples, p_bernoulli):
       
        cross_enropy = (
            samples * tf.math.log(tf.clip_by_value(p_bernoulli, 1e-6, 1e6)) + 
            (1 - samples) * tf.math.log(1 - tf.clip_by_value(p_bernoulli, 1e-6, 1e6))
            )
        
        return cross_enropy


    
    @staticmethod
    def kl_bernoulli_loss(p_1, p_2):
        # Using the full KL divergence formula for Bernoulli distributions
        kl_loss = (
            p_1 * tf.math.log(tf.clip_by_value(p_1 / p_2, 1e-6, 1e6)) + 
            (1 - p_1) * tf.math.log(tf.clip_by_value((1 - p_1) / (1 - p_2), 1e-6, 1e6))
        )
        
        # Handle case when p_1 = 0 (the prior)
        # In this case, the KL reduces to: log(1/(1-p_2))
        is_zero = tf.cast(tf.equal(p_1, 0.0), tf.float32)
        zero_case = tf.math.log(tf.clip_by_value(1 / (1 - p_2), 1e-6, 1e6))
        
        # Combine both cases
        return is_zero * zero_case + (1 - is_zero) * kl_loss
        

    def update_kl_weight(self):
        """Update KL weight according to annealing schedule"""
        self.step.assign_add(1)
        # Sigmoid annealing schedule
        new_weight = self.initial_kl_weight + (self.max_kl_weight - self.initial_kl_weight) * \
                     (1 / (1 + tf.exp(-self.annealing_rate * (tf.cast(self.step, tf.float32) - 1000))))
        self.current_kl_weight.assign(new_weight)
        return self.current_kl_weight


    def __call__(self, predictions, data, gamma):
        # Update KL weight 
        kl_weight = self.update_kl_weight()
        
        intra_loss = (
            tf.reduce_sum(
            tf.stop_gradient(gamma) * 
            self.cross_entropy_loss(data, predictions), axis=None
        )
        )
        
        inter_loss = tf.reduce_sum(
            (1 - tf.stop_gradient(gamma)) * 
            self.kl_bernoulli_loss(self.prior, predictions),
            
            axis=None)
        
        # Apply annealed weighting to the KL term
        total_loss = - intra_loss + kl_weight * inter_loss
        
        # Print current KL weight every 100 steps
        if self.step % 100 == 0:
            tf.print("Current KL weight:", kl_weight)
    
        return total_loss

        
