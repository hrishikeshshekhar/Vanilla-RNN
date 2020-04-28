## Requirements

- numpy
- matplotlib

## Setup

### Install python packages
```
    pip install -r requirements.txt
```

###  Download glove word2vecs
```
wget http://nlp.stanford.edu/data/glove.6B.zip
``` 

### Download imbd reviews data
```
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf imbd_reviews.tar.gz
mv aclImdb imbd_data
``` 

## TODO

- [X] Create a requirements.txt (Add install instructions)
- [X] Make RNN class a separate file
- [X] Display model params 
- [X] Feed forwarding batches and train on mini batches
- [X] Read glove WordToVec
- [X] Save and load weights
- [X] Interactive command line for testing
- [X] Experiment with better initializers (Xavier Initializer)
- [X] Experiment with different optimizers (SGD, Batch GD, Momentum, RMSProp)
- [X] Extract, analyse and train on IMBD Reviews
- [X] Conduct study for hyper parameter tuning
- [X] Compare results with Tensorflow's RNN, LSTM module
- [ ] Check for vanishing and exploding gradients
- [ ] Add regularization (Dropout, L1, L2)
- [ ] Create API for sentiment classification
- [ ] GPU support
