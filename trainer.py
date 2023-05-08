from abc import ABC, abstractclassmethod
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


from util import bitflip_noisy_static, ami_score
from static_dataloader import BATCH_SIZE

class Trainer(ABC):
    """basic rnn-em training class"""
    def __init__(self, em_cell, 
                 loss, 
                 learning_rate):
        self.now = None
        self.loss = loss
        self.em_cell = em_cell
        self.checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                 ami = tf.Variable(-1e10),
                                 optimizer=Adam(learning_rate),
                                 model=self.em_cell.model)
    @abstractclassmethod
    def train_step():
        """implment the iterative EM for some step number"""
    @abstractclassmethod
    def evaluate():
        """ evalutate the AMI score on validation data set"""
    @abstractclassmethod
    def train():
        """train the rnn-em by going trhough data set one time"""

class SequentialTrainer(Trainer):
    """training rnn-em model for sequential type data"""
    #TODO
    pass


class StaticTrainer(Trainer):
    """training rnn-emm model for static type data

    :param Trainer: _description_
    :type Trainer: _type_
    """
    def __init__(self, em_cell,
                 loss, 
                 learning_rate, 
                 checkpoint_dir='./ckpt/static'):
        
        super().__init__(em_cell, loss, learning_rate)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)
        self.restore()

    @property
    def model(self):
        return self.checkpoint.model
    
    def train(self, train_data, valid_data, epoch, evaluate_every=25):
        train_loss_mean = Mean()
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        self.now = time.perf_counter()
        for features, groups in train_data:

            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss_rnn_em, gamma = self.train_step(features)  
            train_loss_mean(loss_rnn_em)
            
                    
            if step % evaluate_every == 0:
                print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
                tloss = train_loss_mean.result()
                train_loss_mean.reset_state() 
                ami_value = self.evaluate(valid_data)
                duration = time.perf_counter() - now
                train_string = f"training for one batch : loss={tloss.numpy():.4f}| ami_score={ami_value.numpy():.4f}| duration={duration:.2f}s"
                print(train_string)
                if ami_value <= ckpt.ami:
                    now = time.perf_counter()
                    continue
                ckpt.ami = ami_value
                ckpt_mgr.save()
                now = time.perf_counter()
    
    @tf.function
    def train_step(self, features, n_iterations=20, K=3):
        features_corrupted = bitflip_noisy_static(features)
        hidden_state = self.em_cell.initial_state(BATCH_SIZE, K)
        for i in range(n_iterations):
            with tf.GradientTape(persistent=True) as tape:
                inputs = (features_corrupted, features) 
                # E-step : computing gammas
                hidden_state  = self.em_cell(inputs, hidden_state)
                rnn_state, preds, gamma = hidden_state
                loss_rnn_em  = self.loss(preds, features, gamma) 
            # M-step : maximizing EM loss interpolated with kl loss 
            gradients = tape.gradient(loss_rnn_em, self.checkpoint.model.trainable_weights)
            self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_weights))
        return loss_rnn_em, gamma
    
    @tf.function
    def evaluate(self, dataset, K=3):
        ami_values = []
        for features, groups in dataset:
            hidden_state = self.em_cell.initial_state(BATCH_SIZE, K)
            features_corrupted = bitflip_noisy_static(features)
            inputs = (features_corrupted, features)
            hidden_state  = self.em_cell(inputs, hidden_state)
            rnn_state, preds, gamma = hidden_state
            # loss_rnn_em  = self.loss(preds, features, gamma)
            ami_valid = ami_score(gamma, groups)
            ami_values.append(ami_valid)
            return tf.reduce_mean(ami_values)
        
    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"model restored from checkpoint at step {self.checkpoint.step.numpy()}.")
