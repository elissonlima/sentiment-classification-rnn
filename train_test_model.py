import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_squared_error
from rnn import RNN


def build_dataset(sequence_len):   

    sin_wave = np.array([math.sin(x) for x in range(200)])

    #training data
    X = []
    Y = []
    num_records = len(sin_wave) - sequence_len # 150

    # X entries are 50 data points
    # Y entries are the 51st data point

    for i in range(num_records - 50):
        X.append(sin_wave[i:i+sequence_len])
        Y.append(sin_wave[i+sequence_len])

    X = np.expand_dims(np.array(X), axis=2) # 100 x 50 x 1
    Y = np.expand_dims(np.array(Y), axis=1) # 100 x 1


    # validation data
    X_validation = []
    Y_validation = []
    for i in range(num_records-sequence_len, num_records):
        X_validation.append(sin_wave[i:i+sequence_len])
        Y_validation.append(sin_wave[i+sequence_len])
    
    X_validation = np.expand_dims(np.array(X_validation), axis=2)
    Y_validation = np.expand_dims(np.array(Y_validation), axis=1)

    return X, Y, X_validation, Y_validation


if __name__ == '__main__':

    sequence_len = 50
    hidden_dim = 100
    output_dim = 1
    epochs = 25

    X, Y, X_validation, Y_validation = build_dataset(sequence_len)

    model = RNN(sequence_len, hidden_dim, output_dim)
    model.fit(X,Y, epochs, 
        X_validation=X_validation, Y_validation=Y_validation)

    # predictions on the training set
    predictions = model.predict(X)
    
    plt.plot(predictions[:, 0,0], 'g')
    plt.plot(Y[:, 0], 'r')
    plt.title("Training Data Predictions in Green, Actual in Red")
    plt.show()