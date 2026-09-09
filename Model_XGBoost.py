import numpy as np
from xgboost import XGBClassifier
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


def Model_XGBoost(train_data, test_data, EP=None, BS=None, neighbors=5):
    print('Model_Xgboost')
    if EP is None:
        EP = 5
    if BS is None:
        BS = 4
    Targets = Targeted_sequence(np.concatenate((train_data, test_data), axis=0))
    train_target = Targets[:train_data.shape[0], :]
    test_target = Targets[train_data.shape[0]:, :]

    IMG_SIZE = 10
    Train_Temp = np.zeros((train_data.shape[0], IMG_SIZE))
    for i in range(train_data.shape[0]):
        Train_Temp[i, :] = np.resize(train_data[i], IMG_SIZE)
    train_data = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE)

    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], IMG_SIZE)
    test_data = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE)

    model = XGBClassifier(learning_rate=0.01, objective="binary:logistic", random_state=42, eval_metric="auc")

    num_samples = train_data.shape[0]
    num_batches = int(np.ceil(num_samples / BS))

    for epoch in range(EP):
        print(f"Epoch {epoch + 1}/{EP}")
        for batch in range(num_batches):
            start = batch * BS
            end = min(start + BS, num_samples)
            batch_data = train_data[start:end]
            batch_target = train_target[start:end]
            model.fit(batch_data, batch_target)

    # Make predictions
    pred = np.zeros(test_target.shape)
    for i in range(test_target.shape[1]):
        model.fit(train_data.tolist(), train_target[:, i].tolist())
        Y_pred = model.predict(test_data.tolist())
        pred[:, i] = np.asarray(Y_pred)

    Eval = Evaluation(test_target, pred)
    return Eval, pred
