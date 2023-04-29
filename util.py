import tensorflow as tf 

def corrupted_data(x):

    # Assume x is a continuous tensor with shape 
    # (batch_size, height, width, channels)
    p = 0.2 # probability of bit flip
    mask = tf.cast(
        tf.random.uniform(tf.shape(x)) > p, 
        dtype=x.dtype
        )
    noisy_x = x * mask

    return noisy_x 

