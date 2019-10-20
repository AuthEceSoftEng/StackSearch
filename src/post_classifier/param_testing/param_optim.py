import gc
import numpy as np
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import uniform, choice, quniform
from keras import backend
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from clsf_test import retain_top_words


def data():
    np.random.seed(1200)  # for reproducibility
    max_features = 12000
    max_len = 4000

    # load the train dataset
    X_train = np.load('train_posts_vec.npy')
    X_test = np.load('test_posts_vec.npy')
    y_train = np.load('train_labels.npy')
    y_test = np.load('test_labels.npy')

    X_train = retain_top_words(X_train, max_features)
    X_test = retain_top_words(X_test, max_features)

    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_train, y_test, max_len, max_features

def model(X_train, X_test, y_train, y_test, max_len, max_features):
    embed_len = {{choice(list(range(15, 65, 5)))}}
    nkernel = {{choice(list(range(2, 7)))}}
    npool = {{choice(list(range(2, 7)))}}
    drop = {{choice([round(x * 0.01, 2) for x in range(10, 80, 5)])}}
    rdrop = {{choice([round(y * 0.01, 2) for y in range(10, 80, 5)])}}
    nepoch = {{choice(list(range(4, 10)))}}
    print(embed_len, nkernel, npool, drop, rdrop, nepoch)

    model = Sequential()
    model.add(Embedding(max_features, embed_len, input_length=max_len))
    model.add(Conv1D(filters=embed_len, kernel_size=nkernel, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=npool))
    model.add(LSTM(100, dropout=drop, recurrent_dropout=rdrop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model on the train dataset
    model.fit(X_train, y_train, epochs=nepoch, batch_size=100)
    # Final evaluation of the model
    score, acc = model.evaluate(X_test, y_test)

    print('Test score:', score, '__ Test accuracy:', acc * 100)
    # Cleaning up used memory in the tensorflow backend
    backend.clear_session()
    gc.collect()
    return {'loss': -acc, 'status': STATUS_OK}#, 'model': model}

if __name__ == '__main__':
    '''best_run, best_model'''
    best_run = optim.minimize(model=model,
                              data=data,
                              algo=tpe.suggest,
                              max_evals=350000,
                              trials=Trials())
    print(best_run)
