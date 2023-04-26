import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np

import rnn_em_cell as rnn_em
from q_graph import Q_graph
from dataloader import get_dataset, generator, BATCH_SIZE, SEQUENCE_LENGHT
from util import corrupted_data
from loss import em_loss

K = 3 
lr = 1e-4
optimizer = keras.optimizers.Adam(learning_rate=lr)
max_epoch = 100 
inner_cell =  Q_graph()
rnn_cell = rnn_em(inner_cell, input_shape=(24, 24, 1))
train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")


gammas = []
for epoch in range(max_epoch):
    for step, (features, groups) in enumerate(train_data):
        features_corrupted = corrupted_data(features)
        hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
        # h, pred, gamma = hidden_state
        batch_loss = .0
        with tf.GradientTape() as tape:
            for i in range(SEQUENCE_LENGHT):

                inputs = features_corrupted[i], features[i+1] 
                hidden_state  = rnn_cell(inputs, hidden_state)
                h, pred, gamma = hidden_state

                gammas.append(gamma)
                loss_rnn_em  = em_loss(pred, features[i+1], gamma)
                batch_loss += loss_rnn_em
        
        gradients = tape.gradient(batch_loss, rnn_cell.trainable_weights)
        optimizer.apply_gradients(zip(gradients, rnn_cell.trainable_weights)) 
        
        gammas = tf.stack(gammas) # suitable size to be checked 
        
        ami = adjusted_mutual_info_score(groups[:-1], gammas.numpy())
             
    if step % 25 == 0:
        print(f"Training loss : {batch_loss:.4f} at batch {step:02d}. Training AMI: {ami:.4f}")


#TODO: Validation loop



