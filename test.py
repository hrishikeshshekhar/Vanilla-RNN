from rnn import RNN 
from data import toy_data, word2vec, embedding_dim

# Loading toy data
train_data = toy_data.train_data
test_data  = toy_data.test_data
output_dim = 2
sentence_length = 10
hidden_dim = 256

# Creating an rnn
rnn = RNN(word2vec, embedding_dim, output_dim, sentence_length, hidden_dim=hidden_dim)

# Loading the weights
save_path = "./weights/weight_data_30_256_0.001.pkl"
rnn.load_weights(save_path)

while(1):
    data = (raw_input("Enter your text : "))
    try:
        preds = rnn.predict(data)
        print("Positive : {}\t Negative : {}".format(preds[1], preds[0]))
    except:
        words = data.split(' ')
        for word in words:
            if(not(word in word2vec.keys())):
                print(" The word '{}' is not there in the dictionary used to train this model.").format(word)
