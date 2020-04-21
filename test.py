from rnn import RNN 
from data import toy_data, word2vec, embedding_dim

# Loading toy data
train_data = toy_data.train_data
test_data  = toy_data.test_data

# Creating an rnn
rnn = RNN(word2vec, embedding_dim, 2)

# Loading the weights
save_path = "./weights/weight_data_30_64.pkl"
rnn.load_weights(save_path)

while(1):
    data = (raw_input("Enter your text : ")).lower()
    try:
        preds, _ = rnn.predict(data)
        print("Positive : {}\t Negative : {}".format(preds[1], preds[0]))
    except:
        words = data.split(' ')
        for word in words:
            if(not(word in word2vec.keys())):
                print(" The word '{}' is not there in the dictionary used to train this model.").format(word)
