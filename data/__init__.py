import toy_data
import numpy as np
import io

# Helper function to load glove word2vec
def loadGloveModel(gloveFile):
    f = io.open(gloveFile,'r',encoding = "utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("{} words loaded!").format(len(model))
    return model

# Loading the gloveWordToVec
embedding_dim = 50
glove_path = "/home/hrishi/Desktop/Personal/Machine-Learning/Datasets/Word2Vecs/glove.6B." + str(embedding_dim) + "d.txt"
word2vec = loadGloveModel(glove_path)
