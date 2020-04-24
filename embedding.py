import string
import numpy as np
import io

# Manages all conversions of words to text


class embeddings:
    def __init__(self, sentence_length, embedding_dim=50):
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim

        glove_path = "/home/hrishi/Desktop/Personal/Machine-Learning/Datasets/Word2Vecs/glove.6B." + \
            str(embedding_dim) + "d.txt"
        self.word2vec = self.loadGloveModel(glove_path)
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
        print("{} words loaded!").format(len(model))

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
        # Splitting into difference sentences
        words = []
        for sentence in input_sentence.split('.'):
            # Removing all other punctuation from a sentence
            sentence = self.remove_punctuation(sentence)
            for word in sentence.split(' '):
                if(len(word) > 0):
                    words.append(word.lower())

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
                # print("Word {} with index {} doesn't exist in glove word to vec").format(
                #     word, index)
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
        return output_data
