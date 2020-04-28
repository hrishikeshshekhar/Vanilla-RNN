import tensorflow as tf
import numpy as np


class SentimentAnalyser(tf.keras.Model):
    def __init__(self, input_dim, output_dim, sentence_length, optimizer='adam', hidden_dim=64, learning_rate=0.001):
        super(SentimentAnalyser, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sentence_length = sentence_length
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(self.sentence_length, self.input_dim)))
        self.model.add(tf.keras.layers.LSTM(self.hidden_dim))
        self.model.add(tf.keras.layers.Dense(self.output_dim))
        self.model.add(tf.keras.layers.Softmax(axis=1))

        self.model.compile(
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer = optimizer,
            metrics = ['accuracy']
        )

    # Shape of inputs : (batch_size, sentence_length, input_dim)
    def call(self, inputs):
        # Shape of LSTM output : (batch_size, sentence_length, hidden_dim)
        return self.model.predict(inputs)

    # Defining a function to train the network
    def train(self, train_X, train_Y, test_X, test_Y, epochs = 100, batch_size = 256):
        self.model.fit(train_X, train_Y, epochs = epochs,
                       batch_size = batch_size, validation_data = (test_X, test_Y))
        score, acc=self.model.evaluate(test_X, test_Y, batch_size = batch_size)
        print("Score and accuracy are : {} and  {}".format(score, acc))

    def summary(self):
        self.model.summary()
    
    def load_weights(self, save_path):
        self.model.load_weights(save_path)
