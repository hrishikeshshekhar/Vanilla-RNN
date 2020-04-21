import numpy as np
import matplotlib.pyplot as plt

from rnn import RNN 
from data import toy_data, word2vec, embedding_dim

train_data = toy_data.train_data
test_data  = toy_data.test_data

# Creating an rnn
rnn = RNN(word2vec, embedding_dim, 2)

# Displaying a summary of the model
rnn.summary()

# Loading data
# save_path = "./weights/weight_data.pkl"
# rnn.load_weights(save_path)

# Training the rnn
epochs = 10001
losses, correct_values = rnn.train(train_data, test_data, epochs, verbose=True)

# Saving the weights
save_path = "./weights/weight_data.pkl"
rnn.save_weights(save_path)

# Plotting training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross entropy loss")
plt.show()

# Plotting correct classification percentages
plt.plot(correct_values)
plt.xlabel("Epoch")
plt.ylabel("Number of correct values")
plt.show()

# Testing the RNN by forwarding sentences
preds1, _ = rnn.predict("i am good right now")
preds2, _ = rnn.predict("this was very good earlier")
preds3, _ = rnn.predict("i am very bad")
preds4, _ = rnn.predict("this is very sad")

print(preds1, preds2, preds3, preds4)
