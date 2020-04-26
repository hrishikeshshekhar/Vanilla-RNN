import os
import numpy as np
import matplotlib.pyplot as plt
import string
import pprint

class IMBD_data:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        # Paths to training and testing data
        self.training_data_path = "train/"
        self.testing_data_path = "test/"

        # Paths for positive and negative reviews
        self.pos_dir = "pos/"
        self.neg_dir = "neg/"

        # Saving paths to training and testing data
        self.train_pos_dir = self.dataset_path + self.training_data_path + self.pos_dir
        self.train_neg_dir = self.dataset_path + self.training_data_path + self.neg_dir
        self.test_pos_dir = self.dataset_path + self.testing_data_path + self.pos_dir
        self.test_neg_dir = self.dataset_path + self.testing_data_path + self.neg_dir

        # Storing an array of lengths of reviews
        self.pprint = pprint.PrettyPrinter(indent=5)
        self.lengths = []

    def load_training_data(self, num_files=5000):
        # Getting all training_data file names
        train_pos_files = os.listdir(self.train_pos_dir)
        train_neg_files = os.listdir(self.train_neg_dir)

        # Reading each file and appending to lists
        train_X = list()
        train_Y = list()

        # Reading equal amount of positive and negative data

        # Reading positive reviews
        file_indices = np.arange(num_files)
        np.random.shuffle(file_indices)
        for index in file_indices:
            filename = train_pos_files[index]
            filepath = self.train_pos_dir + filename
            with open(filepath, 'r') as train_file:
                review = train_file.read()
                train_X.append(review)
                train_Y.append(True)
                self.lengths.append(len(review.split(' ')))

        # Reading the negative sentences
        for index in file_indices:
            filename = train_neg_files[index]
            filepath = self.train_neg_dir + filename
            with open(filepath, 'r') as train_file:
                review = train_file.read()
                train_X.append(review)
                train_Y.append(False)
                self.lengths.append(len(review.split(' ')))

        return train_X, train_Y

    def load_testing_data(self, num_files=500):
        # Getting all training_data file names
        test_pos_files = os.listdir(self.test_pos_dir)
        test_neg_files = os.listdir(self.test_neg_dir)

        # Reading each file and appending to lists
        test_X = list()
        test_Y = list()

        # Reading equal amount of positive and negative data

        # Reading positive reviews
        file_indices = np.arange(num_files)
        np.random.shuffle(file_indices)
        for index in file_indices:
            filename = test_pos_files[index]
            filepath = self.test_pos_dir + filename
            with open(filepath, 'r') as test_file:
                review = test_file.read()
                test_X.append(review)
                test_Y.append(True)
                self.lengths.append(len(review.split(' ')))

        # Reading the negative sentences
        for index in file_indices:
            filename = test_neg_files[index]
            filepath = self.test_neg_dir + filename
            with open(filepath, 'r') as test_file:
                review = test_file.read()
                test_X.append(review)
                test_Y.append(False)
                self.lengths.append(len(review.split(' ')))

        return test_X, test_Y

    def head(self, num_reviews=10):
        reviews = []
        train_pos_files = os.listdir(self.train_pos_dir)
        for index, filename in enumerate(train_pos_files):
            filepath = self.train_pos_dir + filename
            with open(filepath, 'r') as train_file:
                review = train_file.read()
                reviews.append(review)
            if(index > num_reviews):
                break
        self.pprint.pprint(reviews)

    def display_metrics(self):
        # Evaluating average length and variance of reviews
        avg_length = np.mean(self.lengths)
        max_length = np.max(self.lengths)
        variance = np.mean((self.lengths - avg_length) ** 2)
        print("The average length of a sentence in the dataset is : {}").format(
            avg_length)
        print("The variance of the sentence length is : {}").format(variance)

        counts = [0 for _ in range(max_length + 1)]
        for length in self.lengths:
            counts[length] += 1

        plt.plot(counts)
        plt.xlabel("Sentence length for IMBD reviews")
        plt.ylabel("Count of sentences")
        plt.show()
