import numpy as np
import pickle
import string
import sys
import math


class RNN:
    def __init__(self, input_dim, output_dim, sentence_length, hidden_dim=64, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.sentence_length = sentence_length

        # Calculating xavier initializer constants
        whx = math.sqrt(6) / math.sqrt(self.hidden_dim + self.input_dim)
        whh = math.sqrt(6) / math.sqrt(self.hidden_dim + self.hidden_dim)
        wyh = math.sqrt(6) / math.sqrt(self.output_dim + self.hidden_dim)

        # Creating the initial weights
        self.Whx = np.random.uniform(-whx, whx, (hidden_dim, input_dim))
        self.Whh = np.random.uniform(-whh, whh, (hidden_dim, hidden_dim))
        self.Wyh = np.random.uniform(-wyh, wyh, (output_dim, hidden_dim))

        # Creating the initial biases
        self.bh = np.random.normal(0, 1, (hidden_dim, 1))
        self.by = np.random.normal(0, 1, (output_dim, 1))

    # Performs feed forward on the RNN
    def forward(self, data):
        # Getting dimensions from the input
        try:
            assert(data.shape[0] == self.sentence_length)
            assert(data.shape[1] == self.input_dim)
        except:
            raise ValueError("Expected data with dims of : {} but got data with dims : {}").format(
                (self.sentence_length, self.input_dim, None), data.shape)

        batch_size = data.shape[2]

        # Fixing the initial state as a zero vector
        h = np.zeros((self.hidden_dim,  batch_size))

        # Storing all states
        states = [h]
        for words in data:
            assert(np.array(words).shape == (self.input_dim, batch_size))
            h = np.tanh(np.matmul(self.Whh, h) +
                        np.matmul(self.Whx, words) + self.bh)
            states.append(h)

        # Shape of states : (sentence_length + 1, self.hidden_dim, batch_size)
        states = np.array(states)

        # Calculating the output using state
        # Shape (self.output_dim, batch_size)
        output = np.matmul(self.Wyh, states[-1]) + self.by
        output = np.array(self.softmax(output)).reshape(
            (self.output_dim, batch_size))

        return output, states

    # Calculates softmax of the outputs
    def softmax(self, data):
        # Clipping the data
        np.clip(data, -200, 200, out=data)

        # Calculating softmax
        data = np.exp(data)
        data /= np.sum(data, axis=0)
        return data

    # Performs back propagation
    def backPropagate(self, train_X, train_Y, preds, states, batch_size):
        # Shape (self.output_dim, batch_size)
        dL_dY = preds.T
        for index, pred in enumerate(dL_dY):
            pred[train_Y[index]] -= 1
        dL_dY = dL_dY.T

        # Timesteps
        T = len(states) - 1

        # Calculating gradients for Wyh and by
        # Shape (self.output_dim, self.hidden_dim)
        dL_dWyh = np.matmul(dL_dY, states[-1].T)
        dL_dby = np.sum(dL_dY, axis=1).reshape(self.output_dim, 1)

        # Calculating gradients for Whx, Whh, bh
        dL_dWhh = np.zeros(shape=self.Whh.shape)
        dL_dWhx = np.zeros(shape=self.Whx.shape)
        dL_dbh = np.zeros(shape=self.bh.shape)

        # Calculating dL / dh
        # Shape(self.hidden_dim, batch_size)
        dL_dh = np.matmul(self.Wyh.T, dL_dY)

        for t in reversed(range(T)):
            dL_dWhh += np.matmul(dL_dh *
                                 (1 - (states[t + 1] ** 2)), states[t].T)
            dL_dWhx += np.matmul(dL_dh *
                                 (1 - (states[t + 1] ** 2)), train_X[t].T)

            dL_dbh += np.sum(dL_dh * (1 - (states[t + 1] ** 2)),
                             axis=1).reshape(self.hidden_dim, 1)

            # Updating dL_dh
            dL_dh = np.matmul(self.Whh, dL_dh * (1 - states[t + 1] ** 2))

        # Clipping the gradients for exploding gradients
        for updates in [dL_dWhh, dL_dWhx, dL_dbh, dL_dWyh, dL_dby]:
            updates *= self.learning_rate
            np.clip(updates, -1, 1, out=updates)

        # Updating the weights and biases
        self.Whh -= dL_dWhh
        self.Whx -= dL_dWhx
        self.Wyh -= dL_dWyh
        self.bh -= dL_dbh
        self.by -= dL_dby

    def train(self, train_X, train_Y, test_X, test_Y, epochs, verbose=False, batch_size=32):
        # Checking params
        train_X = np.array(train_X)
        train_Y = [int(target) for target in train_Y]
        test_X = np.array(test_X)
        test_Y = [int(target) for target in test_Y]

        # Checking that train_X and train_Y have equal data points
        try:
            assert(train_X.shape[0] == len(train_Y))
        except:
            print("Number of training samples in train_X and train_Y are not equal. Number of samples in train_X is {} and {} in train_Y").format(
                train_X.shape[0], len(train_Y))

        # Checking that test_X and test_Y have equal data points
        try:
            assert(test_X.shape[0] == len(test_Y))
        except:
            print("Number of training samples in train_X and train_Y are not equal. Number of samples in train_X is {} and {} in train_Y").format(
                test_X.shape[0], len(test_Y))

        # Calculating training and testing data size
        training_size = train_X.shape[0]
        testing_size = test_X.shape[0]

        # Conforming batch size to a maximum of training_size / 2
        batch_size = min(batch_size, training_size / 2)

        # Checking that the dimensions of the training data are correct
        try:
            assert(train_X.shape[1] == self.sentence_length)
            assert(train_X.shape[2] == self.input_dim)
        except:
            print("Expected training data shape to be {} but got {}").format(
                training_size, self.sentence_length, self.input_dim)

        # Checking that the dimensions of the testing data are correct
        try:
            assert(test_X.shape[1] == self.sentence_length)
            assert(test_X.shape[2] == self.input_dim)
        except:
            print("Expected testing data shape to be {} but got {}").format(
                testing_size, self.sentence_length, self.input_dim)

        # Transposing the testing data
        test_X = np.transpose(test_X, (1, 2, 0))

        # Array to store metrics
        losses = []
        correct_ans = []
        log_frequency = max(int(float(epochs) / 100), 1)

        # Converting training and testing data into numpy arrays
        print("Size of training data : {}").format(
            sys.getsizeof(train_X) + sys.getsizeof(train_Y))
        print("Size of testing data : {}").format(
            sys.getsizeof(test_X) + sys.getsizeof(test_Y))

        # Splitting the training data into batches of size batchsize
        batches = int(math.ceil(float(training_size) / batch_size))
        batch_training_X = []
        batch_training_Y = []

        # Splitting training data into batches
        for batch in range(batches):
            start_index = batch_size * batch
            end_index = min(batch_size * (batch + 1), training_size)
            temp_batch_size = end_index - start_index
            batch_train_X = train_X[start_index: end_index]

            # Creating an np array of size (batch_size, max_batch_length, self.hidden_dim)
            print("Sentence length : {}").format(self.sentence_length)
            batch_train_X = np.array(batch_train_X).reshape(
                (temp_batch_size, self.sentence_length, self.input_dim))
            batch_train_Y = np.array(train_Y[start_index: end_index])
            batch_training_X.append(batch_train_X)
            batch_training_Y.append(batch_train_Y)

        # Checking if the correct number of batches were inserted
        assert(len(batch_training_X) == batches)

        # Deleting variables to clear RAM
        del train_X, train_Y

        # Training the net
        for epoch in range(epochs):
            loss = 0
            num_correct = 0

            # Iterating through each training batch
            for batch in range(batches):
                # Picking out one batch of training samples
                train_X = batch_training_X[batch]
                train_Y = batch_training_Y[batch]

                # Train_X from (batch_size, sentence_length, self.input_dim) to (sentence_length, self.input_dim, batch_size)
                train_X = np.transpose(train_X, (1, 2, 0))

                # Feed Forwarding
                preds, states = self.forward(train_X)

                loss -= np.sum(np.log([pred[train_Y[index]] for index, pred in enumerate(preds.T)]))
                num_correct += np.sum(np.argmax(preds, axis=0) == train_Y)

                # Back propagating the error
                self.backPropagate(train_X, train_Y, preds, states, batch_size)

            # Appending loss to training data
            losses.append(loss)
            correct_ans.append((float(num_correct) / training_size) * 100)

            # Printing loss and number of correctly classified values
            if(verbose and epoch % log_frequency == 0):

                print("\n\n\n Epoch : {} \n").format(epoch)
                print("TRAINING_DATA")
                print("Loss : {}").format(loss / training_size)
                print("Correctly classified : {} percent of data").format(
                    100 * float(num_correct) / training_size)

                # Resetting loss and correct answers
                loss = 0
                num_correct = 0

                # Feed Forwarding
                preds, states = self.forward(test_X)

                # Calculating the loss and correct classifications
                loss -= np.sum(np.log([pred[test_Y[index]] for index, pred in enumerate(preds.T)]))
                num_correct += np.sum(np.argmax(preds, axis=0) == test_Y)

                print("                            ")
                print("TESTING_DATA")
                print("Loss : {}").format(loss / testing_size)
                print("Correctly classified : {} percent of data").format(
                    100 * float(num_correct) / testing_size)

        return losses, correct_ans

    def summary(self):
        total_params = (self.hidden_dim * self.hidden_dim) + (self.input_dim * self.hidden_dim) + (
            self.output_dim * self.hidden_dim) + (self.hidden_dim) + (self.output_dim)
        print("\n ====================================================")
        print(" Total trainable parameters : {}".format(total_params))
        print(" Learning Rate : {}".format(self.learning_rate))
        print(" Input dimension : {}".format(self.input_dim))
        print(" Output dimension : {}".format(self.output_dim))
        print(" Hidden dimension : {}".format(self.hidden_dim))
        print("\n ====================================================")
        print("                                                    ")

    def save_weights(self, save_path):
        weights = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'sentence_length': self.sentence_length,
            'Whh': self.Whh,
            'Whx': self.Whx,
            'Wyh': self.Wyh,
            'bh': self.bh,
            'by': self.by
        }

        weights_file = open(save_path, "w")
        pickle.dump(weights, weights_file)

        print("Saved weights to path : {} successfully".format(save_path))

    def load_weights(self, save_path):
        weights_file = open(save_path, "r")
        weights = pickle.load(weights_file)

        # Checking whether model created has input and output dims the same as the loaded file
        if(self.input_dim != weights['input_dim'] or self.hidden_dim != weights['hidden_dim'] or self.output_dim != weights['output_dim']):
            print("Warning : The dimensions of your current model and the loaded model do not match")
            print("Your model dimensions : \n")
            print("Input Dimension : {}").format(self.input_dim)
            print("Hidden Dimension : {}").format(self.hidden_dim)
            print("Output Dimension : {} \n").format(self.output_dim)

            print(" Loaded model's dimensions\n")
            print("Input Dimension : {}").format(weights['input_dim'])
            print("Hidden Dimension : {}").format(weights['hidden_dim'])
            print("Output Dimension : {} \n").format(weights['output_dim'])

            text = raw_input(" Enter [Y] to continue to load model : \t")

            if(text != "Y" and text != "y"):
                print("                        ")
                print("Aborting model loading")
                sys.exit()

        # Checking if the sentence length of loaded model matches the current model
        if(weights['sentence_length'] != self.sentence_length):
            print("Warning : The sentence length of your current model and the loaded model do not match")
            
            text = raw_input(" Enter [Y] to continue to load model : \t")

            if(text != "Y" and text != "y"):
                print("                        ")
                print("Aborting model loading")
                sys.exit()

        # Resassigning weights to correct locations
        self.Whh = weights['Whh']
        self.Whx = weights['Whx']
        self.Wyh = weights['Wyh']
        self.bh = weights['bh']
        self.by = weights['by']
        self.input_dim = weights['input_dim']
        self.output_dim = weights['output_dim']
        self.hidden_dim = weights['hidden_dim']
        self.sentence_length = weights['sentence_length']

        print("Loaded weights from path : {} successfully".format(save_path))

    def predict(self, words):
        words = np.array(words)
        data_size = words.shape[0]

        try:
            assert(words.shape[1] == self.sentence_length)
            assert(words.shape[2] == self.input_dim)
        except:
            raise ValueError("Expected dimesion of input as {} but received dimensions {}").format(
                (data_size, self.sentence_length, self.input_dim), words.shape)

        inputs = words.reshape(data_size, self.sentence_length, self.input_dim)
        inputs = np.transpose(inputs, (1, 2, 0))
        preds, _ = self.forward(inputs)
        return preds
