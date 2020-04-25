import os
import numpy as np
import matplotlib.pyplot as plt
import string

# Reading all the training_data
current_dir = "/home/hrishi/Desktop/Personal/Machine-Learning/RNN_From_Scratch/data/"
training_data_path = "imbd_reviews/train/"
testing_data_path = "imbd_reviews/test/"

# Extracting positive sentiment data
pos_dir = "pos/"
neg_dir = "neg/"

# Getting all training_data and storing it into a csv file
train_pos_files = os.listdir(current_dir + training_data_path + pos_dir)
train_neg_files = os.listdir(current_dir + training_data_path + neg_dir)
test_pos_files = os.listdir(current_dir + testing_data_path + pos_dir)
test_neg_files = os.listdir(current_dir + testing_data_path + neg_dir)

# Reading each file and appending to a data dictionary
test_data = {}
train_data = {}
training_data_X = []
training_data_Y = []
testing_data_X = []
testing_data_Y = []

num_files = 10000
lengths = []
avg_length = 0

# Reading the training data

# Reading positive sentences
for filename in train_pos_files[:num_files]:
    filepath = current_dir + training_data_path + pos_dir + filename
    with open(filepath, 'r') as train_file:
        data = train_file.read()
        lengths.append(len(data.split(' ')))
        train_data[data] = True

# Reading the negative sentences
for filename in train_neg_files[:num_files]:
    filepath = current_dir + training_data_path + neg_dir + filename
    with open(filepath, 'r') as train_file:
        data = train_file.read()
        lengths.append(len(data.split(' ')))
        train_data[data] = False

# Reading the testing data

# Reading positive sentences
for filename in test_pos_files[:num_files]:
    filepath = current_dir + testing_data_path + pos_dir + filename
    with open(filepath, 'r') as test_file:
        data = test_file.read()
        lengths.append(len(data.split(' ')))
        test_data[data] = True

# Reading negative sentences
for filename in test_neg_files[:num_files]:
    filepath = current_dir + testing_data_path + neg_dir + filename
    with open(filepath, 'r') as test_file:
        data = test_file.read()
        lengths.append(len(data.split(' ')))
        test_data[data] = False


# Evaluating a few metrics on the data
lengths = np.array(lengths)
avg_length = np.mean(lengths)
variance = np.mean((lengths - avg_length) ** 2)
print("The average length of a sentence in the dataset is : {}").format(avg_length)
print("The variance of the sentence length is : {}").format(variance)

max_length = np.max(lengths)
counts = [0 for _ in range(max_length + 1)]
for length in lengths:
    counts[length] += 1

plt.plot(counts)
plt.show()
