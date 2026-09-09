import numpy as np
from keras import layers, models
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


def Model_AA_RRNN(trainX, testX, EP=None, BS=None, HN=None, sol=None):
    if sol is None:
        sol = [128, 0.001, 5]
    if BS is None:
        BS = 32
    if EP is None:
        EP = int(sol[2])
    if HN is None:
        HN = int(sol[0])

    Targets = Targeted_sequence(np.concatenate((trainX, testX), axis=0))
    trainY = Targets[:trainX.shape[0], :]
    testY = Targets[trainX.shape[0]:, :]

    input_shape = (5, 32, 32, 3)
    num_classes = testY.shape[-1]
    IMG_SIZE = 32

    # Data Reshaping
    Train_X = np.zeros((trainX.shape[0], 5, IMG_SIZE, IMG_SIZE, 3))
    for i in range(trainX.shape[0]):
        temp = np.resize(trainX[i], (5, IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (5, IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((testX.shape[0], 5, IMG_SIZE, IMG_SIZE, 3))
    for i in range(testX.shape[0]):
        temp = np.resize(testX[i], (5, IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (5, IMG_SIZE, IMG_SIZE, 3))

    # AA-RRNN Model
    inputs = layers.Input(shape=input_shape)

    # ----- CNN Feature Extractor -----
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = layers.TimeDistributed(
        layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(
        layers.Flatten())(x)

    # ----- First LSTM (return sequences for attention) -----
    lstm_out = layers.LSTM(HN, return_sequences=True)(x)

    # ----- Residual Connection -----
    residual = layers.LSTM(HN, return_sequences=True)(lstm_out)
    residual_out = layers.Add()([lstm_out, residual])  # Residual block

    # ----- Attention Mechanism -----
    attention = layers.Attention()([residual_out, residual_out])
    attention = layers.GlobalAveragePooling1D()(attention)

    # ----- Adaptive Dense Layer -----
    dense = layers.Dense(64, activation='relu')(attention)
    outputs = layers.Dense(num_classes, activation='softmax')(dense)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=sol[1]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        Train_X, trainY,
        epochs=EP,
        batch_size=BS,
        verbose=1,
        validation_data=(Test_X, testY)
    )
    pred = model.predict(Test_X)
    Eval = Evaluation(testY, pred)
    return Eval, pred
