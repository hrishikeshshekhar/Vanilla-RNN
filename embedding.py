import string
import numpy as np
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

# Manages all conversions of words to text


class embeddings:
    def __init__(self, sentence_length, embedding_dim=50, remove_stop_words=False):
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.remove_stop_words = remove_stop_words

        glove_path = "/home/hrishi/Desktop/Personal/Machine-Learning/Datasets/Word2Vecs/glove.6B." + \
            str(embedding_dim) + "d.txt"
        self.word2vec = self.loadGloveModel(glove_path)
        self.stopwords = set(stopwords.words('english'))
        self.outlier_words = []

    # Helper function to load glove word2vec
    def loadGloveModel(self, gloveFile):
        f = io.open(gloveFile, 'r', encoding="utf-8")
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("{} words loaded!".format(len(model)))

        return model

    def remove_punctuation(self, sentence):
        output = ""
        for symbol in sentence:
            # Replacing all special symbols with a space 
            if(symbol in string.punctuation):
                output += " "
            else:
                output +=  symbol
        return output

    def tokenize(self, input_sentence):
        # Removing all other punctuation from a sentence
        sentence = self.remove_punctuation(input_sentence)

        # Splitting into difference sentences
        words = word_tokenize(sentence)

        # Removing all the stop words
        words = [word.lower() for word in words]
        
        # Removing stop words
        if(self.remove_stop_words):
            new_words = []
            for word in words:
                if(word not in self.stopwords):
                    new_words.append(word)
            return new_words
        else:
            return words
        
    def pad_sentence(self, words):
        padding = [0 for _ in range(self.embedding_dim)]

        # Padding sentences if size is less than self.sentence_length
        sentence_length = len(words)
        if(sentence_length < self.sentence_length):
            for _ in range(self.sentence_length - sentence_length):
                words.append(padding)

        return words

    def trim_sentence(self, words):
        # Trimming sentences if the length is greater than self.sentence_length
        if(len(words) > self.sentence_length):
            return words[:self.sentence_length]
        return words

    def create_input(self, sentence):
        # Cleaning the inputs
        words = self.tokenize(sentence)
        inputs = []

        for index, word in enumerate(words):
            try:
                inputs.append(self.word2vec[word])
            except:
                # print("The word {} doesn't exist in the word2vec dict".format(word))
                self.outlier_words.append(word)
                inputs.append(np.zeros(self.embedding_dim))

        # Padding the inputs
        padded_inputs = self.pad_sentence(inputs)
        trimmed_inputs = self.trim_sentence(padded_inputs)
        inputs = np.array(trimmed_inputs).reshape(
            self.sentence_length, self.embedding_dim)
        return inputs

    def get_data_from_list(self, sentences):
        '''
            Input list of strings : shape (dataset)
            Output list of word2vecs : shape (data_size, sentence_length, embedding_dim)
        '''
        data_size = len(sentences)
        output_data = np.array([self.create_input(sentence)
                                for sentence in sentences])
        assert(output_data.shape == (
            data_size, self.sentence_length, self.embedding_dim))
        return np.array(output_data)
