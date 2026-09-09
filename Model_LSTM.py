from keras import Sequential
import numpy as np
from keras.src.layers import LSTM, Dense
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


def Model_LSTM(trainX, testX, BS=None, EP=None):
    print('LSTM')
    if BS is None:
        BS = 32
    if EP is None:
        EP = 10
    Targets = Targeted_sequence(np.concatenate((trainX, testX), axis=0))
    trainY = Targets[:trainX.shape[0], :]
    testy = Targets[trainX.shape[0]:, :]

    IMG_SIZE = [1, 100]

    num_classes = testy.shape[-1]

    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((testX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testX.shape[0]):
        Test_Temp[i, :] = np.resize(testX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(Train_X.shape[1], Train_X.shape[-1])))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, trainY, epochs=EP, steps_per_epoch=5, batch_size=BS, verbose=1,
              validation_data=(Test_X, testy))
    pred = model.predict(Test_X, verbose=2)
    pred = np.asarray(pred)
    Eval = Evaluation(testy, pred)
    return Eval, pred

