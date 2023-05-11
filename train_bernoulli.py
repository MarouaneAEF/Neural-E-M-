import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import numpy as np
import matplotlib.pyplot as plt 
import datetime


from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from static_dataloader import get_dataset, generator, BATCH_SIZE
from util import bitflip_noisy_static, ami_score
from bernoulli_loss import em_loss

K = 3 
lr = .0001
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

@tf.function
def validation_step(features):
    hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
    features_corrupted = bitflip_noisy_static(features)
    inputs = (features_corrupted, features)
    hidden_state  = rnn_cell(inputs, hidden_state)
    rnn_state, preds, gamma = hidden_state
    loss_rnn_em  = loss_fn(preds, features, gamma)
    return loss_rnn_em, gamma


# # # # # # # # # # # #
#   Trainning loop    #
# # # # # # # # # # # #
train_ami_mean = Mean()
valid_ami_mean = Mean()
train_loss_mean = Mean()
valid_loss_mean = Mean()
n_iterations = 20
best_valid_score = -1e10
patience = 0 
continue_training = True

for epoch in range(50):
    for step, (features, groups) in enumerate(train_data):
        loss_rnn_em, gamma = train_step(features)  
        # printing out informations 
        train_loss_mean(loss_rnn_em)
        ami_train = ami_score(gamma, groups)
        train_ami_mean(ami_train)         
        if step % 25 == 0:
            print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
            tloss = train_loss_mean.result()
            tami_score = train_ami_mean.result()
            train_string = f"training for one batch : loss={tloss:.4f}| ami_score={tami_score:.4f}"
            print(train_string)
            train_loss_mean.reset_state() 
            train_ami_mean.reset_state()

    for step, (features, groups) in enumerate(valid_data):
        loss_rnn_em, gamma = validation_step(features)
        ami_valid = ami_score(gamma, groups)
        valid_ami_mean(ami_valid)
        valid_loss_mean(loss_rnn_em)

        if step % 10 == 0:
            print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
            vloss = valid_loss_mean.result()
            vami_score = valid_ami_mean.result()
            validation_string = f"validation for one batch: loss={vloss:.4f}| ami_score ={vami_score:.4f}| patience={patience:02d}"
            print(validation_string)

            
    # # Display metrics at the end of each epoch.
    # valid_loss = valid_loss_mean.result()
    # valid_ami_score = valid_ami_mean.result()
    # validation_string = (
    #     f"Validation Epoch--|mean_loss:{valid_loss:.4f}|mean_ami_score: {valid_ami_score:.4f}"
    #                 )
    # print(validation_string)  
            valid_ami_mean.reset_state()   
            valid_loss_mean.reset_state()

            if vami_score > best_valid_score:
                best_valid_score = vami_score
                patience = 0
            else:
                patience += 1 
                if patience >= 250:
                    print("Early stopping!")
                    now = datetime.datetime.now()
                    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
                    model_name = f'./models/rnn_em_model_epoch_{current_time}{epoch}.h5'
                    rnn_cell.model.save_weights(model_name)
                    continue_training = False
                    break

    if not continue_training:
        break    


