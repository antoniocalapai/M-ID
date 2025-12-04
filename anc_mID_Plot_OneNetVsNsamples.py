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

# =============================================
# Setting plotting parameters
sizeMult = 1.5
saveplot = 0
savetable = 0

tickFontSize = 10
labelFontSize = 12
titleFontSize = 14

sns.set(style="whitegrid")
sns.set_context("paper")

# =============================================
# Setting paths
IMG_path = './data/'
PLOT_path = './Plots/'
DATA_path = './Pic_Labels/'
DF_path = './dataframes/'

proportions = list(range(10, 101, 10))
proportions.append(1)
proportions = sorted(proportions)

# =============================================
csv_file = 'OneNet_Balanced_PropPics'
df_name = "{}{}{}".format(DF_path, csv_file, '.csv')
df = pd.read_csv(df_name, index_col=0, low_memory=False)

df = df[(df['predict'] == 1)]

df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
df['hour'] = df['hour'].astype(str).str[:2]
df['hour'] = df['hour'].apply(pd.to_numeric)

#df = df[(df['group'] != 'iggJos')]
df = df.drop(df[(df['group'] == 'iggJos') & (df['session'] >= 4)].index)

df = df.reset_index(drop=True)

# ==
DATA = pd.DataFrame()
for i in proportions:
    T = pd.DataFrame()

    df['correct_' + str(i)] = df['animal'] == df['result_' + str(i)]
    T['total'] = df.groupby(['animal'])['predict'].sum().reset_index()['predict']
    T['correct'] = df.groupby(['animal'])['correct_' + str(i)].sum().reset_index()['correct_' + str(i)] / T['total']
    T['animal'] = df.groupby(['animal'])['correct_' + str(i)].sum().reset_index()['animal']
    T['set_size'] = int(df['CNN_Npics' + str(i)].unique())
    T['Proportion'] = i

    DATA = DATA.append(T)

DATA = DATA.reset_index(drop=True)

# ==== PLOT
figure_height = (150 / 25.4) * sizeMult
figure_width = (180 / 25.4) * sizeMult
fig, ax = plt.subplots(1)

ax = sns.regplot(x="set_size", y="correct", data=DATA, logx=True,
                 scatter_kws={"color": "black", 's': DATA['total'] / 100}, line_kws={"color": "red"})

ax.set_xticks(DATA.set_size.unique())
ax.text(750, 0.77, 'min size = 700 trials', fontsize=12)  # add text
ax.text(750, 0.76, 'max size = 13000 trials', fontsize=12)  # add text

ax.set_xlabel(xlabel='Set Size', fontsize=labelFontSize)
ax.set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)

ax.tick_params(labelsize=tickFontSize)

# plot_name = 'OneNet_EffectSetSize'
# NAME = "{}{}{}".format(PLOT_path, plot_name, '.pdf')
# plt.savefig(NAME, format='pdf')
# NAME = "{}{}{}".format(PLOT_path, plot_name, '.png')
# plt.savefig(NAME, format='png')
