import numpy as np
import matplotlib.pyplot as plt

from rnn import RNN
from embedding import embeddings
from data import IMBD_data

# Defining hyper parameters
learning_rate = 0.001
hidden_dim = 128
batch_size = 128
output_dim = 2
optimizer = "rmsprop"
num_files = 5000
sentence_length = 50
embedding_dim = 50

# Getting the training data
imbd_data = IMBD_data('/home/hrishi/Desktop/Personal/Machine-Learning/RNN_From_Scratch/data/imbd_reviews/')
train_X, train_Y = imbd_data.load_training_data(num_files=num_files)
test_X, test_Y   = imbd_data.load_testing_data(num_files=num_files)

# Creating an embeddings class object
embedding = embeddings(sentence_length, embedding_dim=embedding_dim)

# Creating an rnn
rnn = RNN(embedding_dim, output_dim, sentence_length,
          hidden_dim=hidden_dim, learning_rate=learning_rate)

# Displaying a summary of the model
rnn.summary()

# Loading data
save_path = "./weights/imbd_data/weight_data_" + \
    optimizer + '_' +  str(num_files) + '_' + str(batch_size) + "_" + str(hidden_dim) + "_" + \
    str(learning_rate) + '_' + str(sentence_length) + ".pkl"
try:
    rnn.load_weights(save_path)
except:
    print("No weights exist in path : {}").format(save_path)

# Perparing the input data
train_X = embedding.get_data_from_list(train_X)
test_X = embedding.get_data_from_list(test_X)

# Training the rnn
epochs = 201
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

test_data = embedding.get_data_from_list(tests)
preds = rnn.predict(test_data)
print("Testing data : {}").format(tests)
print("Predictions  : {}").format(preds)
