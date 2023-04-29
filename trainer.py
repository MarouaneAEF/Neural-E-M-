import time
import tensorflow as tf
from tensorflow.python import training
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from tensorflow.keras.metrics import Mean


import rnn_em_cell as rnn_em
from q_graph import Q_graph
from dataloader import get_dataset, generator, BATCH_SIZE, SEQUENCE_LENGHT
from util import corrupted_data
from loss import em_loss

K = 3 
lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
max_epoch = 100 
time_frame = 25
inner_cell =  Q_graph()
rnn_cell = rnn_em(inner_cell, input_shape=(24, 24, 1))
train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")

class Trainer:

    def __init__(self, model=rnn_cell) -> None:

        self.now = None
        @property
        def model(self):
            return self.model 
        
        def train(self,):

            train_loss_mean = Mean()
            train_ami_mean = Mean()
            valid_ami_mean = Mean()
            
            self.now  = time.perf_counter()

            
            for epoch in range(max_epoch):
                start_time = time.time()
                for step, (features, groups) in enumerate(train_data):
                    features_corrupted = corrupted_data(features)
                    hidden_state = self.model.initial_state(BATCH_SIZE, K)
                    # h, pred, gamma = hidden_state
                    
                    loss_value, responsibility = self.train_step(features, hidden_state, features_corrupted)
                    ami_train = adjusted_mutual_info_score(groups[:-1], responsibility)
                    train_loss_mean(loss_value)
                    train_ami_mean(ami_train)
                    if step % time_frame == 0: 
                        
                        duration = time.perf_counter - self.now

                        step_string = (
                             f"Training_step--|step:{step:05d}|loss-value:{loss_value:.4f}|ami_score:{ami_train:.4f}|durations:{duration:.2f}s"
                        )     
                        print(step_string)
                        
                    # Display metrics at the end of each epoch.
                train_acc = train_loss_mean.result()
                train_ami_score = train_ami_mean.result()
                train_string = (
                    f"Mean Training--|mean_loss:{train_acc:.4f}|mean_ami_score: {train_ami_score:.4f}"
                                )
                print(train_string)
                # Reset training metrics at the end of each epoch
                train_loss_mean.reset_state()
                train_ami_mean.reset_state()
                # Now run a validation loop at the end of each epoch.
                responsibilities = []
                for step, (features, groups) in valid_data:
                    features_corrupted = corrupted_data(features)
                    hidden_state = self.model.initial_state(BATCH_SIZE, K)
                    for i in range(SEQUENCE_LENGHT):
                        inputs = features_corrupted[i], features[i+1]
                        hidden_state = self.model(inputs, hidden_state)
                        _, _, responsibility = hidden_state
                        responsibilities.append(responsibility)
            
                    responsibilities = tf.stack(responsibilities)
                    ami_valid = adjusted_mutual_info_score(groups[:-1], responsibilities)
                    valid_ami_mean(ami_valid)
                valid_ami_mean = valid_ami_mean.result() 
                valid_string = f"Validation--|mean_ami_score: {train_ami_score:.4f}"   
                print(valid_string)
                valid_ami_mean.reset_state()        


    @tf.function
    def train_step(self, features, hidden_state, features_corrupted):
        with tf.GradientTape() as tape:
            responsibilities = []
            for i in range(SEQUENCE_LENGHT):

                inputs = features_corrupted[i], features[i+1] 
                hidden_state  = self.model(inputs, hidden_state)
                _, pred, gamma = hidden_state
                responsibilities.append(gamma)
                loss_rnn_em  = em_loss(pred, features[i+1], gamma)
                batch_loss += loss_rnn_em
        
        gradients = tape.gradient(batch_loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_weights)) 
        
        return batch_loss, tf.stack(responsibilities)
    