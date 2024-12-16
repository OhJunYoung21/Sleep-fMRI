import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3.3), strides=1, padding="same", activation='relu', input_shape=(28, 28, 1))
        self.maxpool1 = layers.MaxPool2D((3, 3), strides=2, padding='same')

        self.conv2 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')
        self.maxpool2 = layers.MaxPool2D((3, 3), strides=2, padding='same')

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


cnn = CNN()


