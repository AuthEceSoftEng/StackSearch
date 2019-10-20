import os
import sys
import datetime
from timeit import default_timer as timer

import numpy as np

from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from metrics import f1, precision, recall

## Folders
vec_folder = 'training_data/vec_data/'
model_folder = 'models/'

## Filter out INFO, WARNING logging in Tensorflow backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class PostClassifier:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = load_model(model_path)

    def load_data_from_file(self, data_path):
        return np.load(data_path)

    def retain_top_words(self, data_matrix, ntop_tokens):
        """
        Given a matrix data (list of np.array) retain the n most frequent tokens
        (ntop_tokens) of each row-vector.
        """
        res_matrix = []
        for row in data_matrix:
            if isinstance(row, list):
                res_matrix.append([el for el in row if el < ntop_tokens])
            else:
                res_matrix.append(row[row < ntop_tokens])
        return res_matrix

    def process_vecs(self, data_matrix, ntop_tokens, max_length):
        """
        Given a data matrix retain the n most frequent tokens and pad each
        row-vector to max_length length.
        """
        data_matrix = self.retain_top_words(data_matrix, ntop_tokens)
        data_matrix = sequence.pad_sequences(data_matrix, maxlen=max_length)
        return np.array(data_matrix)

    def feed_data(self,
                  data_matrix,
                  ntop_tokens=12000,
                  max_length=4000,
                  batch_size=6000,
                  verbose=0):
        """
        Utility function to feed data into the prediction system.
        In case the rows of the data matrix is bigger than batch_size the matrix
        is split into batches in order to conserve memory during the prediction
        process.
        """

        def progress(iterable, max_n=30, verbose=0):
            n = len(iterable)
            for index, element in enumerate(iterable):
                if verbose:
                    j = (index + 1) / n
                    print(
                        '\r[{:{}s}] {}%'.format('=' * int(max_n * j), max_n,
                                                int(100 * j)),
                        end='  ')
                    print(str(index + 1) + '/' + str(n), end='  ')
                    stime = timer()
                    yield index, element
                    print(
                        ' ETA:',
                        datetime.timedelta(
                            seconds=((n - index - 1) * (timer() - stime))),
                        end='')
                else:
                    yield index, element
            if verbose:
                print()

        nrows = len(data_matrix)
        n_batches = 1
        if nrows > batch_size:
            n_batches = round(nrows / batch_size)
            print('Batches:', n_batches)
        data_matrix = np.array_split(data_matrix, n_batches)
        for ii, batch in progress(data_matrix, verbose=verbose):
            yield self.process_vecs(batch, ntop_tokens, max_length)

    def make_prediction(self, data_matrix, verbose=1):
        """
        Make a prediction (1 unclean, 0 clean) for every row-vector of the given 
        vectorized data matrix.
        """
        if self.model:
            predictions = self.model.predict(
                data_matrix, batch_size=128, verbose=verbose)
            return predictions.round()
        else:
            raise ValueError('Classifier model path unknown.')

    def save_predictions(self, output_path, predictions):
        predictions = predictions.round()
        np.savetxt(output_path, predictions, fmt='%u')


def load_training_data(vec_path,
                       label_path,
                       ntop_tokens=12000,
                       max_length=4000):
    dmatrix = np.load(vec_path)
    labels = np.load(label_path)
    dmatrix = PostClassifier().retain_top_words(dmatrix, ntop_tokens)
    dmatrix = sequence.pad_sequences(dmatrix, maxlen=max_length)
    return dmatrix, labels


def compile_model(max_features=12000,
                  embed_len=35,
                  nrows=4000,
                  kernel_size=3,
                  pool_size=3,
                  dropout=0.6,
                  recurrent_dropout=0.05,
                  metrics=['accuracy']):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=max_features, output_dim=embed_len, input_length=nrows))
    model.add(
        Conv1D(
            filters=embed_len,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(100, dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer='adagrad', metrics=metrics)
    return model


def train_model(model, vec_path, label_path, epochs=6, batch_size=40):
    X, Y = load_training_data(vec_path, label_path)
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    return model


def keras_evaluation(model,
                     X_train,
                     X_test,
                     y_train,
                     y_test,
                     epochs=6,
                     batch_size=40,
                     verbose=1):
    # Fit the model on the train dataset
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # Final evaluation of the model
    score, acc = model.evaluate(X_test, y_test, verbose=verbose)
    print('Test score:', score, '__ Test accuracy:', acc * 100)


def scikit_evaluation(X,
                      Y,
                      build_fn,
                      epochs=6,
                      batch_size=40,
                      splits=5,
                      verbose=1,
                      seed=1200,
                      shuffle=True,
                      metrics=['accuracy', 'precision', 'f1']):
    model = KerasClassifier(
        build_fn=build_fn,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose)
    kfold = StratifiedKFold(
        n_splits=splits, shuffle=shuffle, random_state=seed)
    for metric in metrics:
        results = cross_val_score(model, X, Y, cv=kfold, scoring=metric)
        print(results.mean())
        print(results)


def parameter_optimization(param_grid, model, X, Y, cv=5, threads=1,
                           verbose=1):
    #param_grid parameter dictionary
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=threads,
        verbose=verbose)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_,
                                 grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == '__main__':
    # fix random seed for reproducibility
    seed = 1200
    np.random.seed(seed)

    #scikit_evaluation(X, Y, compile_model)
    model = compile_model(metrics=['accuracy'])
    model = train_model(model, vec_folder + 'post_vecs_extra.npy',
                        vec_folder + 'labels_extra.npy')
    model.save(model_folder + 'c-lstm_v1.1.hdf5')
