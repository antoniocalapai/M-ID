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

# =============================================
models = ['OneNet_Bal_ExtSetSize', 'MultiNet_UnBal_ExtSetSize',
          'MultiNet_UnBalanced', 'Online_CoreML']
palette = {models[0]: "b", models[1]: "g", models[2]: "red", models[3]: "grey"}
DATA = pd.DataFrame()

for m in models:
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)

    df = df[(df['predict'] == 1)]
    df = df[(df['animal'] == 'igg')]
    df = df[(df['session'] == 8)]

    print(m, len(df[df['train'] == 1]))

    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df['correct'] = df['animal'] == df['result']
    categories = df['animal'].unique()

    T = df.groupby(['animal', 'session', 'hour'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'session', 'hour'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = m

    T = T.sort_values(by=['session', 'animal', 'hour'])
    T = T.reset_index(drop=True)

    DATA = DATA.append(T)

DATA = DATA.reset_index(drop=True)

figure_height = (150 / 25.4) * sizeMult
figure_width = (180 / 25.4) * sizeMult

fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=False, sharey=False,
                       figsize=(figure_width, figure_height), gridspec_kw={'width_ratios': [1, 0.3]})

sns.barplot(x="session", y="accuracy", hue="Model", data=DATA, ci=95, palette=palette, ax=ax[0, 0])
sns.barplot(x="animal", y="accuracy", hue="Model", data=DATA, ci=95, palette=palette, ax=ax[1, 0])
sns.barplot(x="hour", y="accuracy", hue="Model", data=DATA, ci=95, palette=palette, ax=ax[2, 0])

ax[0, 0].set_ylim([0.75, 1.01])
ax[1, 0].set_ylim([0.75, 1.01])
ax[2, 0].set_ylim([0.75, 1.01])

ax[0, 0].set_xlabel(xlabel='Sessions', fontsize=labelFontSize)
ax[0, 0].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)

ax[1, 0].set_xlabel(xlabel='Animals', fontsize=labelFontSize)
ax[1, 0].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)

ax[2, 0].set_xlabel(xlabel='Time of Day', fontsize=labelFontSize)
ax[2, 0].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)

ax[0, 0].tick_params(labelsize=tickFontSize)
ax[1, 0].tick_params(labelsize=tickFontSize)
ax[2, 0].tick_params(labelsize=tickFontSize)

ax[0, 0].legend(prop={'size': 14}, loc='lower left')
ax[1, 0].legend([], [], frameon=False)
ax[2, 0].legend([], [], frameon=False)
