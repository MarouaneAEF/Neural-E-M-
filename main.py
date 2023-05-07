import tensorflow as tf
from tensorflow import keras
import numpy as np
from util import ami_score
import numpy as np

from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from dataloader import get_dataset, generator, BATCH_SIZE, SEQUENCE_LENGHT
from util import corrupted_data
from bernoulli_loss import em_loss

K = 5 
lr = 1e-5
optimizer = keras.optimizers.Adam(learning_rate=lr)
max_epoch = 100 
inner_cell =  Q_graph()
loss_fn = em_loss()
rnn_cell = rnn_em(inner_cell, input_shape=(24, 24, 1))
train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")



for epoch in range(max_epoch):
    for step, (features, groups) in enumerate(train_data):
        features_corrupted = corrupted_data(features)
        hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
        # h, pred, gamma = hidden_state
        batch_loss = .0
        gammas = []
        # A single RNN-EM step is used for each timestep. May be we would better use more
        # thant one step 
        with tf.GradientTape() as tape:
            for i in range(SEQUENCE_LENGHT):

                inputs = features_corrupted[i], features[i+1] 
                hidden_state  = rnn_cell(inputs, hidden_state)
                _, pred, gamma = hidden_state

                gammas.append(gamma)
                loss_rnn_em  = loss_fn(pred, features[i+1], gamma)
                batch_loss += loss_rnn_em
        
        gradients = tape.gradient(batch_loss, rnn_cell.model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, rnn_cell.model.trainable_weights)) 
        
        gammas = tf.stack(gammas) 
        
        ami_train = ami_score(groups[:-SEQUENCE_LENGHT+1], gammas)
             
        if step % 25 == 0:
            print(f"Training loss : {batch_loss:.4f} at batch {step:02d} at epoch{epoch:03d} Training AMI: {ami_train:.4f}")


    # Validation loop
    for step, (features, groups) in valid_data:
        features_corrupted = corrupted_data(features)
        hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
        
        for i in range(SEQUENCE_LENGHT):
            inputs = features_corrupted[i], features[i+1]
            hidden_state = rnn_cell(inputs, hidden_state)
            _, _, gamma = hidden_state
            gammas.append(gamma)
        
        gammas = tf.stack(gammas)
        ami_valid = ami_score(groups[:-1], gammas)
        if step % 25 == 0:
            print(f"Training loss : {batch_loss:.4f} at batch {step:02d}. Validation AMI: {ami_valid:.4f}")





