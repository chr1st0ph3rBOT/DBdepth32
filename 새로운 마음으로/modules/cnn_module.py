# modules/cnn_module.py

import tensorflow as tf
from tensorflow.keras import layers, models

class CNNModel(tf.keras.Model):
    def __init__(self, output_dim=256):
        super(CNNModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation='relu')
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        """
        inputs: (batch, 84, 84, 1)
        """
        x = tf.cast(inputs, tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)
