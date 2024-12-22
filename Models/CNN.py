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
        self.fc2 = layers.Dense(1, activation='sigmoid')

    def build(self, input_shape):
        # 여기서 입력 크기에 따라 레이어를 명시적으로 초기화 가능
        super(CNN, self).build(input_shape)

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

X_train, X_test, y_train, y_test = train_test_split(shen_pkl[feature_name], shen_pkl['STATUS'], test_size=0.2,
                                                    random_state=42)

X_train = np.array([item[0] for item in X_train])
X_test = np.array([item[0] for item in X_test])

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = CNN()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
