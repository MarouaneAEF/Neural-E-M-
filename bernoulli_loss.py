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
        p_clipped = tf.clip_by_value(p_bernoulli, 1e-6, 1.0 - 1e-6)
        cross_enropy = (
            samples * tf.math.log(p_clipped) +
            (1 - samples) * tf.math.log(1.0 - p_clipped)
        )
        return cross_enropy


    
    @staticmethod
    def kl_bernoulli_loss(p_1, p_2):
        eps = 1e-6
        # Clip p_2 directly: bounds the gradient 1/(1-p_2) and log ratios
        p_2 = tf.clip_by_value(p_2, eps, 1.0 - eps)
        # Add eps to p_1 to avoid 0*log(0/p)=NaN when prior=0
        p_1 = tf.clip_by_value(tf.cast(p_1, tf.float32) + eps, eps, 1.0 - eps)
        return (
            p_1 * tf.math.log(p_1 / p_2) +
            (1.0 - p_1) * tf.math.log((1.0 - p_1) / (1.0 - p_2))
        )
        

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
        
        return total_loss

        
