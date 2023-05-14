import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time

from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from static_dataloader import get_dataset, generator, BATCH_SIZE
from util import bitflip_noisy_static, ami_score
from bernoulli_loss import em_loss

K = 3 
lr = .0005
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
max_epoch = 100 
inner_cell =  Q_graph()
loss_fn = em_loss()
rnn_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))

# setting checkpoint
checkpoint_dir = './ckpt/static'
checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                 ami = tf.Variable(-1e10),
                                 optimizer=optimizer,
                                 model=rnn_cell.model)

checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                directory=checkpoint_dir,
                                                max_to_keep=3)

train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")

@tf.function
def train_step(features, n_iterations=20):
    features_corrupted = bitflip_noisy_static(features)
    hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
    for i in range(n_iterations):
        with tf.GradientTape(persistent=True) as tape:
            inputs = (features_corrupted, features) 
            # E-step : computing gammas
            hidden_state  = rnn_cell(inputs, hidden_state)
            rnn_state, preds, gamma = hidden_state
            loss_rnn_em  = loss_fn(preds, features, gamma) 
        # M-step : maximizing EM loss interpolated with kl loss 
        gradients = tape.gradient(loss_rnn_em, rnn_cell.model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, rnn_cell.model.trainable_weights))
    # weight update after n_iterations have been completed
    # E-step : 
    # hidden_state  = rnn_cell(inputs, hidden_state)
    # rnn_state, preds, gamma = hidden_state
    # loss_rnn_em  = loss_fn(preds, features, gamma) 
    # # M-step : 
    # gradients = tape.gradient(loss_rnn_em, rnn_cell.model.trainable_weights)
    # optimizer.apply_gradients(zip(gradients, rnn_cell.model.trainable_weights)) 
    return loss_rnn_em, gamma


def validation(dataset):
    ami_values = []
    hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
    for features, groups in dataset:
        features_corrupted = bitflip_noisy_static(features)
        inputs = (features_corrupted, features)
        hidden_state  = rnn_cell(inputs, hidden_state)
        _, _, gamma = hidden_state
        # loss_rnn_em  = loss_fn(preds, features, gamma)
        ami_valid = ami_score(gamma, groups)
        ami_values.append(ami_valid)

    return tf.reduce_mean(ami_values)
        


# # # # # # # # # # # #
#   Trainning loop    #
# # # # # # # # # # # #
n_iterations = 100

for epoch in range(n_iterations):
    train_ami_mean = Mean()
    train_loss_mean = Mean()
    now = time.perf_counter()
    for step, (features, groups) in enumerate(train_data):
        checkpoint.step.assign_add(1)
        loss_rnn_em, gamma = train_step(features)  
        train_loss_mean(loss_rnn_em)
        ami_train = ami_score(gamma, groups)
        train_ami_mean(ami_train)         
        if step % 25 == 0:
            print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
            tloss = train_loss_mean.result()
            tami_score = train_ami_mean.result()
            train_loss_mean.reset_state() 
            train_ami_mean.reset_state()
            vami_score = validation(valid_data.take(30))
            duration = time.perf_counter() - now
            train_string = f"training for one batch : loss={tloss.numpy():.4f}| t_ami_score={tami_score:.4f} | v_ami_score={vami_score.numpy():.4f} | duration={duration:.2f}s"
            print(train_string)
            if vami_score.numpy() <= checkpoint.ami:
                now = time.perf_counter()
                continue
            checkpoint.ami = vami_score
            checkpoint_manager.save()
            now = time.perf_counter()
