
import tensorflow as tf 

class em_loss(object):

    def __init__(self, prior=0):
        self.prior = prior
        

    @staticmethod
    def cross_entropy_loss(samples, p_bernoulli):
        samples * tf.math.log(tf.clip_by_value(p_bernoulli, 1e-6, 1e6)) + (1 - samples) * tf.math.log(1 - tf.clip_by_value(p_bernoulli, 1e-6, 1e6))
    
    
    @staticmethod
    def kl_bernoulli_loss(p_1, p_2):

        return (
            p_1 * tf.math.log(p_1 / tf.clip_by_value((p_2, 1e-6, 1e6))) + 
                (1 - p_1) * tf.math.log((1 - p_1)/tf.clip_by_value((1 - p_2), 1e-6, 1e6))
                )

    def __call__(self, predictions, data, gamma):

        intra_loss = (
            tf.reduce_sum(tf.stop_gradient(gamma)) * 
            self.cross_entropy_loss(data, predictions)
        )

        inter_loss = (
            tf.reduce_sum(
            (1 - tf.stop_gradient(gamma)) * 
            self.kl_bernoulli_loss(self.prior, predictions)
            )
        )

        total_loss = intra_loss + inter_loss
        return total_loss

        
