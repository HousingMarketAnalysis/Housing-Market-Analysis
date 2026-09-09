import numpy as np
from matplotlib import pylab
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import warnings
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


def Statastical(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_convergence():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'NGO-AA-RRNN', 'DOA-AA-RRNN', 'AOA-AA-RRNN', 'QSO-AA-RRNN', 'LD-QSO-AA-RRNN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv = np.zeros((Fitness.shape[-2], 5))
    for j in range(len(Algorithm) - 1):
        Conv[j, :] = Statastical(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv[j, :])
    print('-------------------------------------------------- Statistical Report ',
          '  --------------------------------------------------')
    print(Table)

    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#0cff0c', linewidth=3, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#cf6275', linewidth=3, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    plt.savefig("./Results/Convergence.png")
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_Batchsize():
    eval = np.load('Eval_all_BS.npy', allow_pickle=True)
    Terms = ['BERTScore', 'BLEU', 'ROUGE', 'METEOR', 'Exact Match(EM)', 'F1 Score', 'Response Time (s)',
             'Mean Reciprocal Rank(MRR)']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]
    Methods = ['LSTM', 'RRNN', 'LD-QSO-AA-RRNN']
    batch_size = ['4', '8', '16', '32', '64']
    for j in range(len(Graph_Terms)):
        Graph = eval[1:, :, Graph_Terms[j]]

        ALG_Val = Graph[:, 7:]
        bar_width = 0.15  # Width of the bars
        X = np.arange(ALG_Val.shape[0])  # Positions for bars
        colour = ['#00BFC4', '#FF9224', '#B5D334', '#9370DB', '#FF6347']
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = []
        for i in range(ALG_Val.shape[-1]):
            bars.append(ax.bar(X + i * bar_width * 1.25, ALG_Val[:, i], color=colour[i], edgecolor='w', width=bar_width,
                               label=Methods[i]))
            ax.bar_label(container=bars[i], size=9, label_type='edge', labels=[f'{x:.3f}' for x in ALG_Val[:, i]],
                         rotation=90, fontweight='bold', padding=5)

        ax.set_xticks(X + (bar_width * 1.25))
        ax.set_xticklabels(batch_size[1:])
        ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
        circle_markers = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colour[i], markersize=10) for i in
                          range(len(Methods))]
        ax.legend(circle_markers, Methods, title="", fontsize=12, loc='upper center',
                  bbox_to_anchor=(0.5, -0.1),
                  frameon=False, ncol=5)
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        ax.spines['left'].set_color('lightgray')
        ax.spines['left'].set_linewidth(0.0)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Batch Size vs ' + Terms[Graph_Terms[j]])
        plt.tight_layout()
        path = "./Results/Batch Size vs %s.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def Plot_Kfold():
    eval = np.load('Eval_all_Fold.npy', allow_pickle=True)
    Terms = ['BERTScore', 'BLEU', 'ROUGE', 'METEOR', 'Exact Match(EM)', 'F1 Score', 'Response Time (s)',
             'Mean Reciprocal Rank(MRR)']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7]

    Algorithms = ['NGO-AA-RRNN', 'DOA-AA-RRNN', 'AOA-AA-RRNN', 'QSO-AA-RRNN', 'LD-QSO-AA-RRNN']
    Classifiers = ['ANN', 'XGBoost', 'LSTM', 'RRNN', 'LD-QSO-AA-RRNN']
    KFOLD = ['1', '2', '3', '4', '5']
    for j in range(len(Graph_Terms)):
        Graph = eval[:5, :, Graph_Terms[j]]
        Alg_Graph = np.array([Graph[:, 0], Graph[:, 1], Graph[:, 2], Graph[:, 3], Graph[:, 4]])
        colors = ['#fe2c54', '#fa5ff7', '#01a049', '#02ccfe', 'black']  # '#c071fe'
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.15
        index = np.arange(len(KFOLD))
        for i, Algorithm in enumerate(Algorithms):
            bars = ax.bar(index + i * bar_width, Alg_Graph[i], bar_width, label=Algorithms[i], color=colors[i])
            ax.bar_label(
                bars,
                labels=[f'{val:.3f}' for val in Alg_Graph[i]],
                padding=3,
                rotation=90,
                fontsize=9,
                # fontweight='bold',
                color='black'
            )
        ax.set_xlabel('K FOLD →', fontsize=12, fontweight='bold', color='#35530a')
        ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold',
                      color='#35530a')
        ax.set_xticks(index + bar_width * (len(Algorithms) / 2 - 0.5))
        ax.set_xticklabels(KFOLD)
        dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                       in colors]
        plt.legend(dot_markers, Algorithms, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10,
                   frameon=False, ncol=3, prop={'weight': 'bold'})
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        path = "./Results/Kfold_%s_Alg.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Terms[j]] + ' of Algorithms')
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        Mtd_Graph = np.array([Graph[:, 5], Graph[:, 6], Graph[:, 7], Graph[:, 8], Graph[:, 4]])
        colors = ['#f97306', '#82a67d', '#bc13fe', '#13eac9', 'black']
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.15
        index = np.arange(len(KFOLD))
        for i, Classifier in enumerate(Classifiers):
            bars = ax.bar(index + i * bar_width, Mtd_Graph[i], bar_width, label=Classifiers[i], color=colors[i])
            ax.bar_label(
                bars,
                labels=[f'{val:.3f}' for val in Mtd_Graph[i]],
                padding=3,
                rotation=90,
                fontsize=9,
                # fontweight='bold',
                color='black'
            )
        ax.set_xlabel('K FOLD →', fontsize=12, fontweight='bold', color='#35530a')
        ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold',
                      color='#35530a')
        ax.set_xticks(index + bar_width * (len(Classifiers) / 2 - 0.5))
        ax.set_xticklabels(KFOLD)
        dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                       in colors]
        plt.legend(dot_markers, Classifiers, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10,
                   frameon=False, ncol=3, prop={'weight': 'bold'})
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        path = "./Results/Kfold_%s_Mtd.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Terms[j]] + ' of Classifier')
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def Plot_Learinng_percentage():
    eval = np.load('Eval_all_LP.npy', allow_pickle=True)
    Terms = ['BERTScore', 'BLEU', 'ROUGE', 'METEOR', 'Exact Match(EM)', 'F1 Score', 'Response Time (s)',
             'Mean Reciprocal Rank(MRR)']
    Table_Term = [0, 1, 2, 3, 4, 5, 6, 7]
    Learning_percentage = ['35 %', '45 %', '55 %', '65 %', '75 %', '85 %']
    Algorithm = ['Learning Percentage', 'NGO-AA-RRNN', 'DOA-AA-RRNN', 'AOA-AA-RRNN', 'QSO-AA-RRNN', 'LD-QSO-AA-RRNN']
    Classifier = ['Learning Percentage', 'ANN', 'XGBoost', 'LSTM', 'RRNN', 'LD-QSO-AA-RRNN']
    for k in range(len(Table_Term)):
        value = eval[:, :, Table_Term[k] ]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Learning_percentage)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j])
        print('--------------------------------------------------', str(Terms[Table_Term[k]]),
              ' Learning percentage vs Algorithm Comparison ' + ' --------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Learning_percentage)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1])
        print('-------------------------------------------------- ', str(Terms[Table_Term[k]]),
              'Learning percentage vs Classifier Comparison ' + ' --------------------------------------------------')
        print(Table)


if __name__ == '__main__':
    plot_convergence()
    Plot_Batchsize()
    Plot_Kfold()
    Plot_Learinng_percentage()

