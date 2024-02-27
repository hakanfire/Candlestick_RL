import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Ekran görüntüsünü alma
def screenshot():
    import pyautogui
    img = pyautogui.screenshot()
    return np.array(img)

# CNN modeli
class CNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# DNN modeli
class DNNModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

# Politika modeli
class PolicyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return tf.nn.softmax(x)

# A3C algoritması
def a3c(env, agent):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.add((state, action, reward, next_state, done))
            state = next_state

        agent.learn()

# Ortam
class Env:
    def __init__(self):
        self.prices = []

    def reset(self):
        self.prices = []
        return self.prices

    def step(self, action):
        # Ekran görüntüsünü al
        img = screenshot()

        # CNN modeli ile mum grafiğini tanı
        cnn_model = CNNModel()
        predictions = cnn_model(img)

        # DNN modeli ile karar ver
        dnn_model = DNNModel()
        action = dnn_model(predictions)

        # Gelecek barı tahmin et
        next_price = self.prices[-1] * (1 + np.random.uniform(-0.05, 0.05))

        # Ödülü hesapla
        reward = 0
        if action == 0 and next_price > self.prices[-1]:
            reward = 1
        elif action == 1 and next_price < self.prices[-1]:
            reward = 1

        # Bitti mi kontrol et
        done = False
        if len(self.prices) >= 100:
            done = True

        self.prices.append(next_price)

        return next_price, reward, done, {}

# Ajan
class Agent:
    def __init__(self):
        self.memory = []
        self.epsilon = 0.1
        self.q_model = CNNModel()
        self.policy_model = PolicyModel()
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.update_policy_counter = 0

    def act(self, state):
        # Epsilon-greedy ile seçim yap
        if np.random.rand() < self.epsilon:
            
