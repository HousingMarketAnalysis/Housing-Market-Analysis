import numpy as np
import os
import pandas as pd
from numpy import matlib
import Global_vars
from AOA import AOA
from Chatbot import Chatbot
from DOA import DOA
from Model_AA_RRNN import Model_AA_RRNN
from Model_ANN import Model_ANN
from Model_LSTM import Model_LSTM
from Model_XGBoost import Model_XGBoost
from NGO import NGO
from PROPOSED import PROPOSED
from Plot_Results import *
from QSO import QSO
from objfun import objfun


def convert_to_numeric(df, columns):
    for col in columns:
        unique_values = df[col].unique()
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        df[col] = df[col].map(value_map)
    return df


# Read the dataset
an = 0
if an == 1:
    Datasets = './Dataset/manipulated dataset 2.xlsx'
    df = pd.read_excel(Datasets, header=1)
    columns_to_convert = ['Name of the Respondent', 'Age group of the respondent', 'Occupation Type',
                          'Gender of the respondent', 'Range of Annual Income', 'Work Location', 'Mode of Commute',
                          'What type of residence do you currently live in?', 'Are you living in your own property?',
                          'Would you like to buy a property in the future',
                          'What is your purpose to buy a property?', 'What type of property you would prefer in the future?',
                          'Reason to buy an independent home/standard apartment/gated community/individual villa/duplex triplex',
                          'Kindly select the range of sizes for your property of preference',
                          'What is your ideal home configuration?  [Bed rooms ]',
                          'What is your ideal home configuration?   [Bathrooms ]', 'Preferred Floor level',
                          'Your level  importance for safety of the surrounding ', 'Open well ventilated terrace',
                          'Your level of Importance for privacy in the design and orientation of your property', ]
    Datas = convert_to_numeric(df, columns_to_convert)
    Datas = np.asarray(Datas).astype('float')
    Datas = np.nan_to_num(Datas)
    np.save('Data.npy', Datas)


# Optimization
an = 0
if an == 1:
    Feat = np.load('Data.npy', allow_pickle=True)
    Global_vars.Feat = Feat
    Npop = 10
    Chlen = 3  # hidden neuron count, Learning rate, Epoches in GRU
    xmin = matlib.repmat(np.asarray([5, 0.01, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 0.99, 50]), Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("NGO...")
    [bestfit1, fitness1, bestsol1, time1] = NGO(initsol, fname, xmin, xmax, Max_iter)  # NGO

    print("DOA...")
    [bestfit2, fitness2, bestsol2, time2] = DOA(initsol, fname, xmin, xmax, Max_iter)  # DOA

    print("AOA...")
    [bestfit3, fitness3, bestsol3, time3] = AOA(initsol, fname, xmin, xmax, Max_iter)  # AOA

    print("QSO...")
    [bestfit4, fitness4, bestsol4, time4] = QSO(initsol, fname, xmin, xmax, Max_iter)  # QSO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_cls.npy', np.asarray(BestSol_CLS))  # Save the Best sol


# Classification
an = 0
if an == 1:
    Feat = np.load('Data.npy', allow_pickle=True)
    BestSol = np.load('BestSol_cls.npy', allow_pickle=True)
    k_fold = 5
    Per = 1 / k_fold
    EVAL = []
    Perc = round(Feat.shape[0] * Per)
    for i in range(k_fold):
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Eval = np.zeros((10, 25))
        for j in range(BestSol.shape[0]):
            sol = BestSol[j, :]
            Eval[j, :], pred0 = Model_AA_RRNN(Train_Data, Test_Data, sol=sol)
        Eval[5, :], pred1 = Model_ANN(Train_Data, Test_Data)
        Eval[6, :], pred2 = Model_XGBoost(Train_Data, Test_Data)
        Eval[7, :], pred3 = Model_LSTM(Train_Data, Test_Data)
        Eval[8, :], pred4 = Model_AA_RRNN(Train_Data, Test_Data)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_all_Fold.npy', np.asarray(EVAL))

plot_convergence()
Plot_Batchsize()
Plot_Kfold()
Plot_Learinng_percentage()
Chatbot()

