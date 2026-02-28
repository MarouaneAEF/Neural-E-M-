import tensorflow as tf
from tensorflow.keras import layers


class Q_graph(tf.keras.Model):

    def __init__(self):
        super(Q_graph, self).__init__()
        
        # Even more simplified encoder with fewer parameters - optimized for M3
        self.bloc_encoder = tf.keras.Sequential(
         [   
            layers.LayerNormalization(), 
            # Fix the reshape operation to handle the flattened input (784 = 28*28)
            layers.Reshape((28, 28, 1)),
            # Reduced number of filters and added batch normalization
            layers.Conv2D(
                filters=8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(
                filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            # Removed the third conv layer to simplify the model
            layers.Flatten(),
            # Reduced size of the dense layer
            layers.Dense(128, activation = 'relu'),
            layers.BatchNormalization(),
            layers.Reshape(target_shape = (128, 1)),
            ])
        
        # LSTM replaces SimpleRNN: gating mechanisms prevent vanishing gradients
        # across EM iterations, enabling the cell to track cluster state reliably
        self.rnn = layers.LSTM(64, return_state=True)

        # Simplified decoder with fewer parameters
        self.decoder_bloc = tf.keras.Sequential(
            [
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(7*7*16),
                layers.Reshape(target_shape = (7, 7, 16)),
                # Reduced filters and added batch normalization
                tf.keras.layers.Conv2DTranspose(filters=16, 
                                                kernel_size=3, 
                                                strides=2, 
                                                padding='same',
                                                activation='relu'),
                layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(filters=1, 
                                                kernel_size=3, 
                                                strides=2, 
                                                padding='same',
                                                activation='sigmoid'),
                layers.Flatten(),
                ])

    def call(self, inputs, theta):
        x = self.bloc_encoder(inputs)
        # LSTM returns (output, h_state, c_state); pack both states together
        x, h_state, c_state = self.rnn(x, initial_state=theta)
        theta = [h_state, c_state]
        x = self.decoder_bloc(x)
        return x, theta
