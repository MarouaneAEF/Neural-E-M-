
import tensorflow as tf 

class em_loss(object):

    def __init__(self, prior={"mu":.0, "sigma":.25}):
        self.prior = prior
        

    @staticmethod
    def log_normal_loss(samples, mu, sigma):

        return ( 
            .5 * ((samples - mu) ** 2) / (tf.clip_by_value(sigma ** 2, 1e-6, 1e6) ** 2) 
                + tf.math.log(tf.clip_by_value(sigma ** 2, 1e-6, 1e6))
        )
    @staticmethod
    def kl_normal_loss(mu1, mu2, sigma1, sigma2):

        return .5 * (

         tf.math.log(sigma2 / sigma1) + (mu2 - mu1) ** 2 / (sigma2 ** 2) + sigma1 ** 2 / sigma2 ** 2 - .5
        ) 

    def __call__(self, mu, data, gamma):

        # when we backprop through the loss gamma must be treated as constant dL/dgamma = 0 
        intra_loss = tf.reduce_sum(
            tf.stop_gradient(gamma) * self.log_normal_loss(data, mu, 1.0)
            )
        #  the convergence of this rnn-em is not guaranteed
        #  unlike the pure neural em algorithm
        #  we add the the extra loss to penalize the EM loss 
        # by adding KL divergence between the pixels predictions with gamma_iK = 0 
        # (out-of-cluster) and prior of the pixel (cf. the article)
        inter_loss =  tf.reduce_sum(
            (1 - tf.stop_gradient(gamma)) * self.kl_normal_loss(self.prior["mu"], mu, 1.0, self.prior["sigma"] )
            )
        total_loss = intra_loss + inter_loss
        return total_loss
