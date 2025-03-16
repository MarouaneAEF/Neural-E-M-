from abc import ABC, abstractmethod
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
    @abstractmethod
    def train_step(self, features):
        """implement the iterative EM for some step number"""
        pass
        
    @abstractmethod
    def evaluate(self, dataset):
        """evaluate the AMI score on validation data set"""
        pass
        
    @abstractmethod
    def train(self, train_data, valid_data, epoch):
        """train the rnn-em by going through data set one time"""
        pass

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
    
    def train(self, train_data, valid_data, epoch, evaluate_every=5):
        train_loss_mean = Mean()
        train_ami_mean = Mean()  # Add AMI tracking for training
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        self.now = time.perf_counter()
        
        # Use tqdm for progress tracking if available
        try:
            from tqdm import tqdm
            # Use a reasonable default since n_epochs might not be known here
            n_epochs = getattr(self, 'n_epochs', 100)
            progress_bar = tqdm(train_data, desc=f"Epoch {epoch+1}/{n_epochs}")
        except ImportError:
            progress_bar = train_data
            
        for features, groups in progress_bar:
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss_rnn_em, gamma = self.train_step(features)  
            train_loss_mean(loss_rnn_em)
            
            # Also calculate and track AMI score during training
            ami_train = ami_score(gamma, groups)
            train_ami_mean(ami_train)
            
            if step % evaluate_every == 0:
                print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
                tloss = train_loss_mean.result()
                tami = train_ami_mean.result()
                train_loss_mean.reset_state() 
                train_ami_mean.reset_state()
                ami_value = self.evaluate(valid_data)
                duration = time.perf_counter() - self.now
                train_string = f"training: loss={tloss.numpy():.4f}| train_ami={tami.numpy():.4f}| valid_ami={ami_value.numpy():.4f}| duration={duration:.2f}s"
                print(train_string)
                
                # Update progress bar description if available
                try:
                    progress_bar.set_description(f"Epoch {epoch+1} - Loss: {tloss.numpy():.4f}, AMI: {ami_value.numpy():.4f}")
                except:
                    pass
                    
                if ami_value.numpy() <= ckpt.ami:
                    self.now = time.perf_counter()
                    continue
                    
                ckpt.ami = ami_value
                ckpt_mgr.save()
                print(f"Model checkpoint saved with AMI: {ami_value.numpy():.4f}")
                self.now = time.perf_counter()
    
    @tf.function
    def train_step(self, features, n_iterations=40, K=3):  # Increased from 20/30 to 40
        features_corrupted = bitflip_noisy_static(features)
        hidden_state = self.em_cell.initial_state(BATCH_SIZE, K)
        for i in range(n_iterations):
            with tf.GradientTape(persistent=True) as tape:
                inputs = (features_corrupted, features) 
                # E-step : computing gammas
                hidden_state  = self.em_cell(inputs, hidden_state)
                _, preds, gamma = hidden_state
                loss_rnn_em  = self.loss(preds, features, gamma) 
            # M-step : maximizing EM loss interpolated with kl loss 
            gradients = tape.gradient(loss_rnn_em, self.checkpoint.model.trainable_weights)
            # Apply gradient clipping to prevent exploding gradients
            gradients = [tf.clip_by_norm(g, 3.0) if g is not None else g for g in gradients]
            self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_weights))
        return loss_rnn_em, gamma
    
    
    def evaluate(self, dataset, K=3):
        ami_values = []
        hidden_state = self.em_cell.initial_state(BATCH_SIZE, K)
        for features, groups in dataset:
            features_corrupted = bitflip_noisy_static(features)
            inputs = (features_corrupted, features)
            hidden_state  = self.em_cell(inputs, hidden_state)
            _, _, gamma = hidden_state
            # loss_rnn_em  = self.loss(preds, features, gamma)
            ami_valid = ami_score(gamma, groups)
            ami_values.append(ami_valid)
        return tf.reduce_mean(ami_values)
        
    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f"model restored from checkpoint at step {self.checkpoint.step.numpy()}.")
