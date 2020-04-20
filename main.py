import numpy as np
import matplotlib.pyplot as plt
from data import toy_data

train_data = toy_data.train_data
test_data  = toy_data.test_data

# Creating the vocabulary
vocab = set([word for sentence in train_data for word in sentence.split(' ')])
vocab_size = len(vocab)
# print("Vocab size", vocab_size)

# Creating a mapping of word to index
word_to_index = {}
index_to_word = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index
    index_to_word[index] = word

# Helper function to create input vectors


def createInputs(words):
    inputs = []

    for word in words.split(' '):
        vec = np.zeros(vocab_size)
        vec[word_to_index[word]] = 1
        inputs.append(vec)

    inputs = np.array(inputs).reshape((len(inputs), vocab_size, 1))
    return inputs


input_vec = np.array(createInputs(train_data.keys()[0]))

# Creating the RNN class


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Creating the initial weights
        self.Whx = np.random.normal(0, 1, (hidden_dim, input_dim))
        self.Whh = np.random.normal(0, 1, (hidden_dim, hidden_dim))
        self.Wyh = np.random.normal(0, 1, (output_dim, hidden_dim))

        # Creating the initial biases
        self.bh = np.random.normal(0, 1, (hidden_dim, 1))
        self.by = np.random.normal(0, 1, (output_dim, 1))

    # Performs feed forward on the RNN
    def forward(self, words):
        # Fixing the initial state as a zero vector
        h = np.zeros((self.hidden_dim, 1))

        states = [h]
        for word in words:
            word = np.array(word).reshape((self.input_dim, 1))
            h = np.tanh(np.matmul(self.Whh, h) +
                        np.matmul(self.Whx, word) + self.bh)
            states.append(h)

        # Calculating the output using state
        output = np.matmul(self.Wyh, states[-1]) + self.by
        output = np.array(self.softmax(output)).reshape((self.output_dim))
        states = np.array(states).reshape((len(states), self.hidden_dim, 1))

        return output, states

    # Calculates softmax of the outputs
    def softmax(self, data):
        data = np.exp(data)
        data /= np.sum(data)
        return data

    # Performs back propagation
    def backPropagate(self, inputs, target, preds, states):
        # Building dL / dy
        dL_dY = preds
        dL_dY[target] -= 1

        # Timesteps
        T = len(states) - 1

        # Calculating gradients for Wyh and by
        dL_dY = dL_dY.reshape((self.output_dim, 1))
        dL_dWyh = np.matmul(dL_dY, states[-1].T)
        dL_dby = dL_dY

        # Calculating gradients for Whx, Whh, bh
        dL_dWhh = np.zeros(shape=self.Whh.shape)
        dL_dWhx = np.zeros(shape=self.Whx.shape)
        dL_dbh = np.zeros(shape=self.bh.shape)

        # Calculating dL / dh
        dL_dh = np.matmul(self.Wyh.T, dL_dY)

        for t in reversed(range(T)):
            dL_dWhh += np.matmul(dL_dh *
                                 (1 - (states[t + 1] ** 2)), states[t].T)
            dL_dWhx += np.matmul(dL_dh *
                                 (1 - (states[t + 1] ** 2)), inputs[t].T)
            dL_dbh += dL_dh * (1 - (states[t + 1] ** 2))

            # Updating dL_dh
            dL_dh = np.matmul(self.Whh, dL_dh * (1 - states[t + 1] ** 2))

        # Clipping the gradients for exploding gradients
        for updates in [dL_dWhh, dL_dWhx, dL_dbh, dL_dWyh, dL_dby]:
            np.clip(updates, -1, 1, out=updates)

        # Updating the weights and biases
        self.Whh -= self.learning_rate * dL_dWhh
        self.Whx -= self.learning_rate * dL_dWhx
        self.Wyh -= self.learning_rate * dL_dWyh
        self.bh -= self.learning_rate * dL_dbh
        self.by -= self.learning_rate * dL_dby

    def train(self, training_data, testing_data, epochs):
        losses = []
        correct_ans = []

        for epoch in range(epochs):
            loss = 0
            num_correct = 0

            for line, sent in train_data.items():
                inp = createInputs(line)
                target = int(sent)

                # Feed Forwarding
                preds, states = self.forward(inp)

                # Calculating the loss and correct classifications
                loss -= np.log(preds[target])
                num_correct += int(np.argmax(preds) == target)

                # Back propagating the error
                self.backPropagate(inp, target, preds, states)

            # Appending loss to training data
            losses.append(loss)
            correct_ans.append(num_correct)

            # Printing loss and number of correctly classified values
            if(epoch % 100 == 0):
                print ("                            ")
                print ("                            ")
                print ("                            ")
                print("Epoch : {}").format(epoch)
                print ("                            ")
                print ("TRAINING_DATA")
                print("Loss : {}").format(loss / (len(train_data)))
                print("Correctly classified : {}").format(100 * float(num_correct) / len(training_data))

                # Resetting loss and correct answers
                loss = 0
                num_correct = 0

                # Testing on testing data
                for words, sent in testing_data.items():
                    inp = createInputs(words)
                    target = int(sent)

                    # Feed Forwarding
                    preds, states = self.forward(inp)

                    # Calculating the loss and correct classifications
                    loss -= np.log(preds[target])
                    num_correct += int(np.argmax(preds) == target)

                print ("                            ")
                print ("TESTING_DATA")
                print("Loss : {}").format(loss / (len(train_data)))
                print("Correctly classified : {}").format(100 * float(num_correct) / len(testing_data))

        return losses, correct_ans


# Initialize the RNN
vanilla_rnn = RNN(vocab_size, 64, 2)
inputs = createInputs('good')
outputs, states = vanilla_rnn.forward(inputs)
print(outputs)

epochs = 10001

# Training the rnn
losses, correct_values = vanilla_rnn.train(train_data, test_data, epochs)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross entropy loss")
plt.show()

plt.plot(correct_values)
plt.xlabel("Epoch")
plt.ylabel("Number of correct values")
plt.show()
