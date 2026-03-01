import tensorflow as tf
from tensorflow.keras import layers


class Q_graph(tf.keras.Model):

    def __init__(self):
        super(Q_graph, self).__init__()

        self.layer_norm = layers.LayerNormalization()

        # Positional encoding injected after reshape, before convolutions.
        # Encoder now receives (B, 28, 28, 3): pixel value + (y, x) normalized coords.
        # This lets the model distinguish spatially separate objects with identical intensities.
        self.conv_encoder = tf.keras.Sequential(
         [
            layers.Conv2D(
                filters=8, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(
                filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Reshape(target_shape=(128, 1)),
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

    @staticmethod
    def _positional_encoding(batch_size):
        """Build a (batch_size, 28, 28, 2) grid of normalized (y, x) coordinates."""
        coords = tf.linspace(0.0, 1.0, 28)
        grid_y, grid_x = tf.meshgrid(coords, coords, indexing='ij')  # (28, 28)
        pos = tf.stack([grid_y, grid_x], axis=-1)                    # (28, 28, 2)
        return tf.tile(pos[tf.newaxis], [batch_size, 1, 1, 1])        # (B, 28, 28, 2)

    def call(self, inputs, theta, training=False):
        batch_size = tf.shape(inputs)[0]
        x = self.layer_norm(inputs, training=training)
        x = tf.reshape(x, [batch_size, 28, 28, 1])
        x = tf.concat([x, self._positional_encoding(batch_size)], axis=-1)  # (B, 28, 28, 3)
        x = self.conv_encoder(x, training=training)
        x, h_state, c_state = self.rnn(x, initial_state=theta)
        theta = [h_state, c_state]
        x = self.decoder_bloc(x, training=training)
        return x, theta
