import numpy as np
import pickle
import sys
import math


class RNN:
    def __init__(self, word2vec, input_dim, output_dim, hidden_dim=64, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.word2vec = word2vec

        # Creating the initial weights
        self.Whx = np.random.normal(0, 1, (hidden_dim, input_dim))
        self.Whh = np.random.normal(0, 1, (hidden_dim, hidden_dim))
        self.Wyh = np.random.normal(0, 1, (output_dim, hidden_dim))

        # Creating the initial biases
        self.bh = np.random.normal(0, 1, (hidden_dim, 1))
        self.by = np.random.normal(0, 1, (output_dim, 1))

    def createInputs(self, words):
        inputs = []

        for word in words.split(' '):
            try:
                vec = self.word2vec[word]
            except:
                print("Word {} dosen't exist in glove word to vec").format(word)
                inputs.append(np.zeros(self.hidden_dim))
            inputs.append(vec)

        return inputs

    # Performs feed forward on the RNN
    def forward(self, data):
        # Getting dimensions from the input
        sentence_length = data.shape[0]
        assert(data.shape[1] == self.input_dim)
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
        dL_dby = np.sum(dL_dY, axis = 1).reshape(self.output_dim, 1)

        # Calculating gradients for Whx, Whh, bh
        dL_dWhh = np.zeros(shape=self.Whh.shape)
        dL_dWhx = np.zeros(shape=self.Whx.shape)
        dL_dbh = np.zeros(shape=self.bh.shape)

        # Calculating dL / dh
        dL_dh = np.matmul(self.Wyh.T, dL_dY) # Shape(self.hidden_dim, batch_size)

        for t in reversed(range(T)):
            dL_dWhh += np.matmul(dL_dh *
                                 (1 - (states[t + 1] ** 2)), states[t].T)
            dL_dWhx += np.matmul(dL_dh *
                                 (1 - (states[t + 1] ** 2)), train_X[t].T)
            
            dL_dbh += np.sum(dL_dh * (1 - (states[t + 1] ** 2)), axis = 1).reshape(self.hidden_dim, 1)

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
        self.bh  -= dL_dbh
        self.by  -= dL_dby

    def train(self, training_data, testing_data, epochs, verbose=False, batch_size=1):
        # Checking params
        assert(batch_size <= len(training_data.values()))

        # Array to store metrics
        losses = []
        correct_ans = []
        log_frequency = max(int(float(epochs) / 100), 1)

        # Calculating training and testing data size
        training_size = len(training_data)
        testing_size = len(testing_data)

        # Converting training and testing data into numpy arrays
        training_X = [self.createInputs(line) for line in training_data.keys()]
        training_Y = [int(target) for target in training_data.values()]
        testing_X = [self.createInputs(line) for line in testing_data.keys()]
        testing_Y = [int(target) for target in testing_data.values()]

        # Splitting the training data into batches of size batchsize
        batches = int(math.ceil(float(training_size) / batch_size))
        batch_training_X = []
        batch_training_Y = []

        # Inserting padding to make all batches of even size
        padding = [0 for _ in range(self.input_dim)]

        for batch in range(batches):
            start_index = batch_size * batch
            end_index = min(batch_size * (batch + 1), training_size)
            temp_batch_size = end_index - start_index
            batch_train_X = training_X[start_index: end_index]
            batch_max_size = np.max([len(sentence)
                                     for sentence in batch_train_X])

            # Making all the sentences the same size by padding zeros
            for sentence in batch_train_X:
                sentence_length = len(sentence)
                if(sentence_length < batch_max_size):
                    for _ in range(batch_max_size - sentence_length):
                        sentence.append(padding)

            # Creating an np array of size (batch_size, max_batch_length, self.hidden_dim)
            batch_train_X = np.array(batch_train_X).reshape(
                (temp_batch_size, batch_max_size, self.input_dim))
            batch_train_Y = np.array(training_Y[start_index: end_index])
            batch_training_X.append(batch_train_X)
            batch_training_Y.append(batch_train_Y)

        # Checking if the correct number of batches were inserted
        assert(len(batch_training_X) == batches)

        # Deleting variables to clear RAM
        del training_X, training_Y
        del training_data

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
                                       
                loss -= np.sum(np.log([pred[train_Y[index]]
                                       for index, pred in enumerate(preds.T)]))
                num_correct += np.sum(np.argmax(preds, axis = 0) == train_Y)

                # Back propagating the error
                self.backPropagate(train_X, train_Y, preds, states, batch_size)

            # Appending loss to training data
            losses.append(loss)
            correct_ans.append((float(num_correct) / training_size) * 100)

            # Printing loss and number of correctly classified values
            if(verbose and epoch % log_frequency == 0):
                print("                            ")
                print("                            ")
                print("                            ")
                print("Epoch : {}").format(epoch)
                print("                            ")
                print("TRAINING_DATA")
                print("Loss : {}").format(loss / training_size)
                print("Correctly classified : {} percent of data").format(
                    100 * float(num_correct) / training_size)

                # Resetting loss and correct answers
                loss = 0
                num_correct = 0

                # Testing on testing data one by one 
                for index in range(testing_size):
                    test_X = np.array(testing_X[index]).reshape(1, len(testing_X[index]), self.input_dim)
                    test_Y = [testing_Y[index]]

                    # Transposing the input data
                    test_X = np.transpose(test_X, (1, 2, 0))

                    # Feed Forwarding
                    preds, states= self.forward(test_X)

                    # Calculating the loss and correct classifications
                    loss -= np.sum(np.log([pred[test_Y[index]]
                                        for index, pred in enumerate(preds.T)]))
                    num_correct += np.sum(np.argmax(preds) == test_Y)

                print("                            ")
                print("TESTING_DATA")
                print("Loss : {}").format(loss / testing_size)
                print("Correctly classified : {} percent of data").format(
                    100 * float(num_correct) / testing_size)

        return losses, correct_ans

    def summary(self):
        total_params = (self.hidden_dim * self.hidden_dim) + (self.input_dim * self.hidden_dim) + (
            self.output_dim * self.hidden_dim) + (self.hidden_dim) + (self.output_dim)
        print("                                                    ")
        print("====================================================")
        print(" Total trainable parameters : {}".format(total_params))
        print(" Learning Rate : {}".format(self.learning_rate))
        print(" Input dimension : {}".format(self.input_dim))
        print(" Output dimension : {}".format(self.output_dim))
        print(" Hidden dimension : {}".format(self.hidden_dim))
        print("====================================================")
        print("                                                    ")

    def save_weights(self, save_path):
        weights = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
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
            print("Warning : The dimensions of your current model and the loaded model do not match with the loaded data")
            print("Your model dimensions : ")
            print("                        ")
            print("Input Dimension : {}").format(self.input_dim)
            print("Hidden Dimension : {}").format(self.hidden_dim)
            print("Output Dimension : {}").format(self.output_dim)
            print("                        ")

            print(" Loaded model's dimensions")
            print("                        ")
            print("Input Dimension : {}").format(weights['input_dim'])
            print("Hidden Dimension : {}").format(weights['hidden_dim'])
            print("Output Dimension : {}").format(weights['output_dim'])
            print("                        ")

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

        print("Loaded weights from path : {} successfully".format(save_path))

    def predict(self, words):
        inputs = self.createInputs(words)
        inputs = np.array(inputs).reshape(1, len(inputs), self.input_dim)
        inputs = np.transpose(inputs, (1, 2, 0))
        return self.forward(inputs)
