from keras import Sequential
from keras.src.layers import Dense
import numpy as np
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import Evaluation
from keras.src.utils import to_categorical


def Targeted_sequence(data):
    if len(data.shape) == 2:
        data = data
    else:
        data = data[:, :, 0, 0]
    unique_values = sorted(list(set(data.flatten())))
    value_to_int = {v: i for i, v in enumerate(unique_values)}
    n_vocab = len(unique_values)
    seq_length = data.shape[1] - 1
    y = []
    for row in data:
        y.append(value_to_int[row[seq_length]])
    y = np.array(y)
    y = to_categorical(y, num_classes=n_vocab)
    return y


def Model_ANN(trainX, testX, BS=None):
    if BS is None:
        BS = 4

    Targets = Targeted_sequence(np.concatenate((trainX, testX), axis=0))
    trainY = Targets[:trainX.shape[0], :]
    testy = Targets[trainX.shape[0]:, :]

    num_classes = trainY.shape[-1]
    IMG_SIZE = 784
    Train_X = np.zeros((trainX.shape[0], IMG_SIZE))
    for i in range(trainX.shape[0]):
        temp = np.resize(trainX[i], IMG_SIZE)
        Train_X[i] = np.reshape(temp, IMG_SIZE)

    Test_X = np.zeros((testX.shape[0], IMG_SIZE))
    for i in range(testX.shape[0]):
        temp = np.resize(testX[i], IMG_SIZE)
        Test_X[i] = np.reshape(temp, IMG_SIZE)

    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy']) # binary_crossentropy sparse_categorical_crossentropy
    model.summary()
    model.fit(Train_X, trainY, epochs=5, batch_size=BS, steps_per_epoch=5, validation_data=(Test_X, testy))
    pred = model.predict(Test_X)
    Eval = Evaluation(testy, pred)
    return Eval, pred
