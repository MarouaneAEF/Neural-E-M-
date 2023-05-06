
import tensorflow as tf 

class em_loss(object):

    def __init__(self, prior=0):
        self.prior = prior
        

    @staticmethod
    def cross_entropy_loss(samples, p_bernoulli):
        # samples = tf.squeeze(samples, axis=-1)
        # p_bernoulli = tf.squeeze(p_bernoulli, axis=-1)
        cross_enropy = (
            samples * tf.math.log(tf.clip_by_value(p_bernoulli, 1e-6, 1e6)) + 
            (1 - samples) * tf.math.log(1 - tf.clip_by_value(p_bernoulli, 1e-6, 1e6))
            )
        
        return cross_enropy


    
    @staticmethod
    def kl_bernoulli_loss(p_1, p_2):
       
       kl_loss = ( p_1 * tf.math.log( tf.clip_by_value((p_1 /p_2), 1e-6, 1e6)) + 
                (1 - p_1) * tf.math.log((1 - p_1)/tf.clip_by_value((1 - p_2), 1e-6, 1e6))
       )
    #    kl_loss = tf.squeeze( kl_loss, axis=-1)
       return kl_loss




    def __call__(self, predictions, data, gamma):

        # print(f"gamma: {(gamma)}")
        # print(f"prediction: {tf.shape(predictions)}")
        
        intra_loss = (
            tf.reduce_sum(
            tf.stop_gradient(gamma) * 
            self.cross_entropy_loss(data, predictions), axis=None
        )
        )
        # print(f"intra_loss: {tf.shape(intra_loss)}")
        # print(f"data: {tf.shape(data)}")
        # print(f"predictions: {tf.shape(predictions)}")
        # print(f"self.cross_entropy_loss(data, predictions): {tf.shape(self.cross_entropy_loss(data, predictions))}")
        # print(f"self.kl_bernoulli_loss(self.prior, predictions){tf.shape(self.kl_bernoulli_loss(self.prior, predictions))}")
        # print(f"gamma: {tf.shape(gamma)}")
        inter_loss = tf.reduce_sum(
            (1 - tf.stop_gradient(gamma)) * 
            self.kl_bernoulli_loss(self.prior, predictions),
            
            axis=None)
        

        total_loss = intra_loss + inter_loss
    
        return total_loss

        
