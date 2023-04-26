
import tensorflow as tf 

class em_loss(object):

    def __init__(self):
        pass

    def log_normal_loss(samples, mu, sigma):

        return ( 
            .5 * ((samples - mu) ** 2) / (sigma ** 2) + tf.math.log(sigma)
        )
    
    def kl_normal_loss(mu1, mu2, sigma1, sigma2):

        return .5 * (

         tf.math.log(sigma2 / sigma1) + (mu2 - mu1) ** 2 / (sigma2 ** 2) + sigma1 ** 2 / sigma2 ** 2 - .5
        ) 

    def __call__(self, mu, data, gamma):

        intra_loss = tf.reduce_sum(
            tf.stop_gradient(gamma) * self.log_normal_loss(data, mu, 1.0)
            )
        inter_loss = tf.reduce_sum(
            (1 - tf.stop_gradient(gamma)) * self.kl_normal_loss(mu, .0, 1.0, 1.0 )
            )
        total_loss = intra_loss + inter_loss
        return total_loss
