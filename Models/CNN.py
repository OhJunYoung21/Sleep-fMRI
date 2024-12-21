import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), strides=1, padding="same", activation='relu', input_shape=(268, 268, 1))
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


shen_pkl = pd.read_pickle("../Dynamic_Feature_Extraction/Shen_features/shen_dynamic_CNN.pkl")

feature_name = "FC"

status_1_data = shen_pkl[shen_pkl['STATUS'] == 1]
status_0_data = shen_pkl[shen_pkl['STATUS'] == 0]
# Select only the REHO and STATUS columns
selected_data_1 = status_1_data[[feature_name, 'STATUS']]
selected_data_0 = status_0_data[[feature_name, 'STATUS']]

x_train, y_train, x_test, y_test = train_test_split(shen_pkl[feature_name], shen_pkl['STATUS'], test_size=0.2)

model = CNN()

model.build(input_shape=(None, 268, 268, 1))

model.summary()
