import numpy as np
import matplotlib.pyplot as plt

from rnn import RNN
from data import toy_data, word2vec, embedding_dim

train_data = toy_data.train_data
test_data = toy_data.test_data

# Defining hyper parameters
learning_rate = 0.001
hidden_dim = 16
batch_size = 1
output_dim = 2
sentence_length = 10

# Creating an rnn
rnn = RNN(word2vec, embedding_dim, output_dim, sentence_length,
          hidden_dim=hidden_dim, learning_rate=learning_rate)

# Displaying a summary of the model
rnn.summary()

# Loading data
save_path = "./weights/weight_data_" + \
    str(batch_size) + "_" + str(hidden_dim) + "_" + str(learning_rate) + ".pkl"
try:
    rnn.load_weights(save_path)
except:
    print("No weights exist in path : {}").format(save_path)

# Training the rnn
epochs = 1001
losses, correct_values = rnn.train(
    train_data, test_data, epochs, verbose=True, batch_size=batch_size)

# Saving the weights
rnn.save_weights(save_path)

# Plotting training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross entropy loss")
plt.show()

# Plotting correct classification percentages
plt.plot(correct_values)
plt.xlabel("Epoch")
plt.ylabel("Correct classification percentage")
plt.show()

# Testing the RNN by forwarding sentences
tests = [
    "Happy",
    "I am very happy",
    "I am very bad",
    "This is very sad"
]

preds = [rnn.predict(test) for test in tests]
