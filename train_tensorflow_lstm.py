from lstm import SentimentAnalyser
from embedding import embeddings
from data.imbd_data import IMBD_data

# Deciding on the model hyper params
learning_rate = 0.001
sentence_length = 50
embedding_dim = 50
hidden_dim = 64
output_dim = 2
batch_size = 256
num_files = 5000
epochs = 200
beta = 0.9
optimizer = "adam"

# Creating a sentiment analyser
model = SentimentAnalyser(embedding_dim, output_dim, sentence_length, optimizer=optimizer,
                          hidden_dim=hidden_dim, learning_rate=learning_rate)

# Printing a model summary
model.summary()

# Creating an embeddings object
embedding = embeddings(
    sentence_length, embedding_dim=embedding_dim, remove_stop_words=True)

# Creating a dataset object
imbd_data = IMBD_data(
    '/home/hrishi/Desktop/Personal/Machine-Learning/RNN_From_Scratch/data/imbd_reviews/')
train_X, train_Y = imbd_data.load_training_data(num_files=num_files)
test_X, test_Y = imbd_data.load_testing_data(num_files=num_files)
train_X = embedding.get_data_from_list(train_X)
test_X = embedding.get_data_from_list(test_X)
train_Y = imbd_data.get_array_from_labels(train_Y)
test_Y = imbd_data.get_array_from_labels(test_Y)

# Loading weights if they exist
save_path = "./weights/imbd_data/lstm_weights/" + \
    str(learning_rate) + '_' + str(optimizer) + \
    '_' + str(hidden_dim) + '_' + str(num_files)
try:
    model.load_weights(save_path)
except:
    print("No weights exist in path {}".format(save_path))

# Training the model
model.train(train_X, train_Y, test_X, test_Y,
            epochs=epochs, batch_size=batch_size)


# Saving the weights
model.save_weights(save_path)
