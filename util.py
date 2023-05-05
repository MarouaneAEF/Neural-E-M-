import tensorflow as tf 
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
def corrupted_data(data):

    # Assume x is a continuous tensor with shape 
    # (batch_size, height, width, channels)
    p = 0.2 # probability of bit flip
    mask = tf.cast(
        tf.random.uniform(tf.shape(data)) > p, 
        dtype=data.dtype
        )
    noisy_data = data * mask

    return noisy_data 

def ami_score(input_tensor, target_tensor, channels_axis=2, depth=3):
    # print(f"input_tensor.numpy(): {input_tensor.numpy().shape}")
    # print(f"target_tensor.numpy(): {target_tensor.numpy().shape}")
    input_indices = np.argmax(input_tensor.numpy(), axis=channels_axis)
    one_hot_input = tf.one_hot(input_indices, depth=depth)
    target_indices = np.squeeze(target_tensor.numpy(), axis=channels_axis)
    one_hot_target = tf.one_hot(target_indices, depth=int(depth))
    amis = adjusted_mutual_info_score(one_hot_input.numpy().ravel(),
                                       one_hot_target.numpy().ravel())
    return amis