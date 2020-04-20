import numpy as np
import pickle
import sys

class RNN:
    def __init__(self, word_to_index, input_dim, output_dim, hidden_dim=64, learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.word_to_index = word_to_index

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
            vec = np.zeros(self.input_dim)
            vec[self.word_to_index[word]] = 1
            inputs.append(vec)

        inputs = np.array(inputs).reshape((len(inputs), self.input_dim, 1))
        return inputs

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

    def train(self, training_data, testing_data, epochs, verbose=False):
        losses = []
        correct_ans = []

        for epoch in range(epochs):
            loss = 0
            num_correct = 0

            for line, sent in training_data.items():
                inp = self.createInputs(line)
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
            if(verbose and epoch % 100 == 0):
                print("                            ")
                print("                            ")
                print("                            ")
                print("Epoch : {}").format(epoch)
                print("                            ")
                print("TRAINING_DATA")
                print("Loss : {}").format(loss / (len(training_data)))
                print("Correctly classified : {} percent of data").format(
                    100 * float(num_correct) / len(training_data))

                # Resetting loss and correct answers
                loss = 0
                num_correct = 0

                # Testing on testing data
                for words, sent in testing_data.items():
                    inp = self.createInputs(words)
                    target = int(sent)

                    # Feed Forwarding
                    preds, states = self.forward(inp)

                    # Calculating the loss and correct classifications
                    loss -= np.log(preds[target])
                    num_correct += int(np.argmax(preds) == target)

                print("                            ")
                print("TESTING_DATA")
                print("Loss : {}").format(loss / (len(testing_data)))
                print("Correctly classified : {} percent of data").format(
                    100 * float(num_correct) / len(testing_data))

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
            'input_dim' : self.input_dim,
            'output_dim' : self.output_dim,
            'hidden_dim' : self.hidden_dim,
            'Whh' : self.Whh,
            'Whx' : self.Whx,
            'Wyh' : self.Wyh,
            'bh'  : self.bh,
            'by'  : self.by
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
        return self.forward(inputs)
