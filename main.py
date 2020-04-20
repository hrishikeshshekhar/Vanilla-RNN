import numpy as np
import matplotlib.pyplot as plt

from rnn import RNN 
from data import toy_data

train_data = toy_data.train_data
test_data  = toy_data.test_data

# Creating the vocabulary
vocab = set([word for sentence in train_data for word in sentence.split(' ')])
vocab_size = len(vocab)
print ("Vocab size: {}").format(vocab_size)

# Creating a mapping of word to index
word_to_index = {}
index_to_word = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index
    index_to_word[index] = word

# Creating an rnn
rnn = RNN(vocab_size, 64, 2, word_to_index)

# Displaying a summary of the model
rnn.summary()

# Loading data
save_path = "./weights/weight_data"
rnn.load_weights(save_path)

# Training the rnn
epochs = 5001
losses, correct_values = rnn.train(train_data, test_data, epochs, verbose=True)

# Saving the weights
save_path = "./weights/weight_data"
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
input1 = rnn.createInputs("i am good right now")
input2 = rnn.createInputs("this was very good earlier")
input3 = rnn.createInputs("i am very bad")
input4 = rnn.createInputs("this is very sad")

preds1, _ = rnn.predict(input1)
preds2, _ = rnn.predict(input2)
preds3, _ = rnn.predict(input3)
preds4, _ = rnn.predict(input4)

print(preds1, preds2, preds3, preds4)
