from rnn import RNN 
from data import toy_data

train_data = toy_data.train_data
test_data  = toy_data.test_data
word_to_index = toy_data.word_to_index
vocab_size = toy_data.vocab_size

# Creating an rnn
rnn = RNN(word_to_index, vocab_size, 2)

# Loading the weights
save_path = "./weights/weight_data.pkl"
rnn.load_weights(save_path)

while(1):
    data = (raw_input("Enter your text : ")).lower()
    try:
        preds, _ = rnn.predict(data)
        print("Positive : {}\t Negative : {}".format(preds[1], preds[0]))
    except:
        words = data.split(' ')
        for word in words:
            if(not(word in word_to_index.keys())):
                print(" The word '{}' is not there in the dictionary used to train this model.").format(word)
