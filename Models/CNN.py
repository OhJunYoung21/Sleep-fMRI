import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold
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


shen_pkl = pd.read_pickle("../Static_Feature_Extraction/Shen_features/shen_268_CNN.pkl")

feature_name = "FC"

status_1_data = shen_pkl[shen_pkl['STATUS'] == 1]
status_0_data = shen_pkl[shen_pkl['STATUS'] == 0]
# Select only the REHO and STATUS columns
selected_data_1 = status_1_data[[feature_name, 'STATUS']]
selected_data_0 = status_0_data[[feature_name, 'STATUS']]

rkf_split_1 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)
rkf_split_0 = RepeatedKFold(n_repeats=10, n_splits=10, random_state=42)

i = 0

accuracy = []

for (train_idx_1, test_idx_1), (train_idx_0, test_idx_0) in zip(
        rkf_split_1.split(selected_data_1),
        rkf_split_0.split(selected_data_0)):
    # 라벨 1 데이터의 훈련/테스트 분리
    train_1 = selected_data_1.iloc[train_idx_1]
    test_1 = selected_data_1.iloc[test_idx_1]

    # 라벨 0 데이터의 훈련/테스트 분리
    train_0 = selected_data_0.iloc[train_idx_0]
    test_0 = selected_data_0.iloc[test_idx_0]

    # 훈련 데이터와 테스트 데이터 결합

    train_data = pd.concat([train_1, train_0], axis=0).reset_index(drop=True)
    test_data = pd.concat([test_1, test_0], axis=0).reset_index(drop=True)

    train_data[feature_name] = [item[0] for item in train_data[feature_name]]
    test_data[feature_name] = [item[0] for item in test_data[feature_name]]

    train_data[feature_name] = train_data[feature_name] / 255.0
    test_data[feature_name] = test_data[feature_name] / 255.0

    print(train_data[feature_name][0].shape)

    '''

    model = CNN()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(train_data[feature_name], train_data['STATUS'], epochs=5, batch_size=16,
                        validation_split=0.2)

    # 테스트 데이터 평가
    test_loss, test_acc = model.evaluate(test_data[feature_name], test_data['STATUS'])

    accuracy.append(test_acc)

    print(f"Test Loss: {test_loss:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    
    '''

print(np.mean(accuracy))
