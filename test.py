from rnn import RNN 
from embedding import embeddings
from data import toy_data
import numpy as np

# Loading toy data
train_data = toy_data.train_data
test_data  = toy_data.test_data
output_dim = 2
sentence_length = 50
embedding_dim = 50
hidden_dim = 128

# Creating an embeddings class
embedding = embeddings(sentence_length, embedding_dim=embedding_dim)

# Creating an rnn
rnn = RNN(embedding_dim, output_dim, sentence_length, hidden_dim=hidden_dim)

# Loading the weights
save_path = "./weights/imbd_data/weight_data_rmsprop_5000_128_128_0.001_50.pkl"
rnn.load_weights(save_path)

while(1):
    text = raw_input("Enter your text : ")
    try:
        inputs = np.array([embedding.create_input(text)])
        preds = rnn.predict(inputs)
        print("Positive : {}\t Negative : {}").format(preds[1], preds[0])
    except:
        words = text.split(' ')
        for word in words:
            if(not(word in embedding.word2vec.keys())):
                print(" The word '{}' is not there in the dictionary used to train this model.").format(word)
