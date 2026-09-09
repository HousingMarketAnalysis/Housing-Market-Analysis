import numpy as np
from Global_vars import Global_vars
from Model_AA_RRNN import Model_AA_RRNN


def objfun(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Eval, pred = Model_AA_RRNN(Train_Data, Test_Data, sol)
            Fitn[i] = 1 / Eval[0]  # 1 / (BERTScore)
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Eval, pred = Model_AA_RRNN(Train_Data, Test_Data, sol)
        Fitn = 1 / Eval[0]  # 1 / (BERTScore)
        return Fitn

