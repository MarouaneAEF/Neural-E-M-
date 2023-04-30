
import tensorflow as tf
from tensorflow.keras import layers


class Q_graph(tf.keras.Model):

    def __init__(self):
        super(Q_graph, self).__init__()
        
        self.bloc_encoder = tf.keras.Sequential(
        [  
            layers.LayerNormalization(), 
            # the size of the first dimension must be inferred from the input tensor 
            layers.Reshape((-1, 24, 24, 1)),
            layers.Conv2D(
                filters=32, kernel_size=4, strides=(2, 2), activation='relu'),
            layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation='relu'),
            layers.Conv2D(
                filters=128, kernel_size=4, strides=(2, 2), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation = 'relu'),
            layers.Reshape(target_shape = (512,1)),

            ])

        self.rnn = layers.SimpleRNN(250, activation="sigmoid", return_state=True)

        self.decoder_bloc = tf.keras.Sequential(
            [
                layers.Dense(512),
                layers.Dense(3*3*128),
                layers.Reshape(target_shape = (3, 3, 128)),
                tf.keras.layers.Conv2DTranspose(filters=64, 
                                                kernel_size=4, 
                                                strides=2, 
                                                padding='same',
                                                activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32, 
                                                kernel_size=4, 
                                                strides=2, 
                                                padding='same',
                                                activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=1, 
                                                kernel_size=4, 
                                                strides=2, 
                                                padding='same',
                                                activation='relu'),
                layers.Flatten(),
                
                ])


    def call(self, inputs, theta):
        x = self.bloc_encoder(inputs)
        x, theta = self.rnn(x, initial_state=theta)
        x = self.decoder_bloc(x)

        return x, theta
