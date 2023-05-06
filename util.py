import tensorflow as tf 
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

def bitflip_noisy_static(data):
    p = tf.constant([.1])
    rando_noise = tf.random.uniform(tf.shape(data), maxval=1)
    mask = tf.cast(tf.math.greater(p, rando_noise), dtype=tf.float32)
    return data * (1 - mask) 

def bitflip_noisy_uniform(data):
    p = tf.constant([.2])
    rando_noise = tf.random.uniform(tf.shape(data), maxval=1)
    mask_noise = tf.random.uniform(tf.shape(data), maxval=1)
    mask = tf.cast(tf.math.greater(p, mask_noise), dtype=tf.float32) 
    noisy_data = mask * rando_noise + (1 - mask) * data
    return noisy_data

def bitflip_noisy(data):
    p = tf.constant([.2])
    rando_noise = tf.random.uniform(tf.shape(data), maxval=1)
    mask = tf.cast(tf.math.greater(p, rando_noise), dtype=tf.float32)
    return data * (1 - mask) 

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