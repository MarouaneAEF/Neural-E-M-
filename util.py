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



def ami_score(predictions, targets, K=3):
    resp_indices = tf.math.argmax(predictions, axis=1)
    predictions = tf.reshape(resp_indices, shape=(-1,))
    targets = tf.reshape(targets, shape=(-1,))
    amis = adjusted_mutual_info_score(predictions.numpy().ravel(), targets.numpy().ravel())
    return amis




