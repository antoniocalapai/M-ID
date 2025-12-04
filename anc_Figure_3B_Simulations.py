from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.stats import ranksums
from scipy.stats import ttest_1samp
from scipy import stats
import os
import glob

# Panel A - All Networks performance
# =============================================
# Setting plotting parameters
plot_name = 'Figure_3B'

sizeMult = 1
saveplot = 0
savetable = 0

tickFontSize = 10
labelFontSize = 12
titleFontSize = 14

sns.set(style="whitegrid")
sns.set_context("paper")

exclude_group = 'iggJos'
exclude_sessions = 4

# =============================================
# Setting paths
IMG_path = './data/'
Figure_path = './Figures/RAW/'
PLOT_path = './Plots/'
DATA_path = './Pic_Labels/'
DF_path = './dataframes/'

# =============================================
models = ['Online_CoreML', 'MultiNet_Balanced', 'OneNet_Balanced', 'OneNet_Bal_ExtSetSize']
models_name = ['CoreML', 'MultiNet', 'OneNet', 'OneNet Extended']

figure_height = (190 / 25.4) * sizeMult
figure_width = (190 / 25.4) * sizeMult
fig, ax = plt.subplots(int(len(models)/2), int(len(models)/2),
                       constrained_layout=True, sharex=True, sharey=True,
                       figsize=(figure_width, figure_height))
ax = ax.flatten()

for idx, m in enumerate(models):
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    set_size = len(df[df['train'] == 1])
    print(m, set_size)

    df = df[(df['predict'] == 1)]
    df = df[(df['session'] > 6)]
    # print(m, len(df[df['predict'] == 1]))

    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    # df = df.drop(df[(df['group'] == exclude_group) & (df['session'] >= exclude_sessions)].index)
    df['correct'] = df['animal'] == df['result']

    df = df.sort_values(by=['group', 'animal'])
    df = df.reset_index(drop=True)

    categories = df['animal'].unique()

    n = len(categories)
    T = np.zeros((n, n))

    for i, a in enumerate(categories):
        temp = df[df['animal'] == a]
        for j, b in enumerate(categories):
            T[i, j] = len(temp[temp['result'] == b]) / len(temp)

    T = np.around(T, decimals=2)
    DATA = pd.DataFrame(data=T, index=[categories], columns=[categories])
    g = sns.heatmap(DATA, linewidths=.5, cmap="rocket_r", annot=True, ax=ax[idx], cbar=False)
    ax[idx].set_title("{}{}{}".format(models_name[idx], ', set size: ', set_size), fontsize=labelFontSize)
    ax[idx].set_ylabel(ylabel=None)
    ax[idx].set_xlabel(xlabel=None)

# =============================================
if saveplot:
    NAME = "{}{}{}".format(PLOT_path, plot_name, '.pdf')
    plt.savefig(NAME, format='pdf')

    NAME = "{}{}{}".format(PLOT_path, plot_name, '.png')
    plt.savefig(NAME, format='png')

    NAME = "{}{}{}".format(Figure_path, plot_name, '.pdf')
    plt.savefig(NAME, format='pdf')

    NAME = "{}{}{}".format(Figure_path, plot_name, '.png')
    plt.savefig(NAME, format='png')

