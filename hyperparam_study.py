import numpy as np
import time
import pprint
import matplotlib.pyplot as plt
import pandas as pd
import os

from rnn import RNN
from data import toy_data, word2vec, embedding_dim

# Creating an instance of the pretty printer class
pp = pprint.PrettyPrinter(width=41)

# Importing the data
train_data = toy_data.train_data
test_data = toy_data.test_data

# Iterating through hyper parameters and saving data and plots
learning_rate_min = 0.001
learning_rate_max = 0.1
learning_rate_steps = 5
batch_size_min = 20
batch_size_max = 40
batch_size_steps = 5
hidden_dim_min = 16
hidden_dim_max = 256
hidden_dim_steps = 10

# Defining non variable hyper parameters
output_dim = 2
epochs = 1001
sentence_length = 10
hyper_params = []

# Total number of models that will be trained
total_models = learning_rate_steps * batch_size_steps * hidden_dim_steps
model_number = 1

# Defining the path
base_path = os.getcwd() +"/study/glove/"

for learning_rate in np.linspace(learning_rate_min, learning_rate_max, learning_rate_steps):
    for batch_size in np.linspace(batch_size_min, batch_size_max, batch_size_steps):
        for hidden_dim in np.linspace(hidden_dim_min, hidden_dim_max, hidden_dim_steps):
            # Flooring hidden dim and batch size
            hidden_dim = int(hidden_dim)
            batch_size = int(batch_size)

            # Creating a folder for the rnn
            folder_path = "learning_rate_" + str(learning_rate) + "_batch_size_" + str(batch_size) + "_hidden_dim_" + str(hidden_dim) + "/"
            if not os.path.exists(base_path + folder_path):
                os.mkdir(base_path + folder_path)

            # Starting if the file doesn't exist
            weights_path = base_path + folder_path + "weights.pkl"
            if not os.path.exists(weights_path):

                # Creating an RNN
                rnn = RNN(word2vec, embedding_dim, output_dim, sentence_length, hidden_dim = hidden_dim, learning_rate = learning_rate)

                # Training the model
                start_time = time.time()
                losses, correct_values = rnn.train(train_data, test_data, epochs, batch_size=batch_size)
                end_time = time.time()
                training_time = end_time - start_time

                # Saving the model weights
                rnn.save_weights(weights_path)

                # Creating a plot of the losses and correct percentage and saving the figures
                fig = plt.figure()
                plt.plot(losses)
                plt.xlabel("Epoch")
                plt.ylabel("Cross entropy loss")
                plt.savefig(base_path + folder_path + "training_loss.png")

                # Plotting correct classification percentages
                fig = plt.figure()
                plt.plot(correct_values)
                plt.xlabel("Epoch")
                plt.ylabel("Correct classification percentage")
                plt.savefig(base_path + folder_path + "classification_percent.png")

                # Saving the data into the dictionary
                param_data = {
                    'learning_rate' : learning_rate, 
                    'batch_size' : batch_size,
                    'hidden_dim' : hidden_dim,
                    'training_time' : training_time,
                    'training_loss' : losses[-1],
                    'correct_percent' : correct_values[-1]
                }

                hyper_params.append(param_data)

                print("\n Finished training model {} / {}  with params :   ").format(model_number, total_models)
                pp.pprint(param_data)

            else:
                print("Skipping file : {}" ).format(weights_path)

            model_number += 1

# Saving the hyper params into a pandas dataframe and saving it
hyper_param_data = pd.DataFrame.from_dict(hyper_params)
hyper_param_data.to_csv(base_path + "hyper_param_master_data")
