import numpy as np
import matplotlib.pyplot as plt

from rnn import RNN
from embedding import embeddings
from data import imbd_data

train_data = imbd_data.train_data
test_data = imbd_data.test_data

# Defining hyper parameters
learning_rate = 0.001
hidden_dim = 64
batch_size = 128
output_dim = 2
num_training_data = 2000
num_testing_data = 100
sentence_length = 40
embedding_dim = 50

# Creating an embeddings class object
embedding = embeddings(sentence_length, embedding_dim=embedding_dim)

# Creating an rnn
rnn = RNN(embedding_dim, output_dim, sentence_length,
          hidden_dim=hidden_dim, learning_rate=learning_rate)

# Displaying a summary of the model
rnn.summary()

# Loading data
save_path = "./weights/weight_data_" + \
    str(batch_size) + "_" + str(hidden_dim) + "_" + \
    str(learning_rate) + '_' + str(sentence_length) + ".pkl"
try:
    rnn.load_weights(save_path)
except:
    print("No weights exist in path : {}").format(save_path)

# Perparing the input data
train_X = embedding.get_data_from_list(train_data.keys()[:num_training_data])
train_Y = train_data.values()[:num_training_data]
test_X = embedding.get_data_from_list(test_data.keys()[:num_testing_data])
test_Y = test_data.values()[:num_testing_data]

# Training the rnn
epochs = 5001
losses, correct_values = rnn.train(
    train_X, train_Y, test_X, test_Y, epochs, verbose=True, batch_size=batch_size)

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
