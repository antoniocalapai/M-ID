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
models = ['MultiNet_SameAsCoreML', 'MultiNet_UnBalanced', 'OneNet_UnBalanced', 'Online_CoreML', 'MultiNet_Bal_ExtendedSetSize']
models_name = ['MultiNet_Bal', 'MultiNet_Unb', 'OneNet', 'CoreML', 'Ker_MultiNet_ext']
palette = {models[0]: "b", models[1]: "g", models[2]: "purple", models[3]: "orange", models[4]: "red"}
DATA = pd.DataFrame()

for idx, m in enumerate(models):
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    print(m, len(df[df['train'] == 1]))

    df = df[(df['session'] > 6)]
    df = df[(df['predict'] == 1)]
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    # df = df[(df['group'] != 'iggJos')]

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

figure_height = (80 / 25.4) * sizeMult
figure_width = (80 / 25.4) * sizeMult

fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey=False, figsize=(figure_width, figure_height))

g = sns.barplot(x="Model", y="accuracy", hue="Model",dodge=False, data=DATA, ci=95, palette=palette, ax=ax[0])

# ax[0].set_ylim([0.90, 1])
ax[0].legend([], [], frameon=False)
ax[0].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)
ax[0].set_xlabel(xlabel=None)
g.legend(loc='center right', bbox_to_anchor=(1.65, 0.2), ncol=1)
g.set(xlabel=None)
# ==
DATA_stat = pd.DataFrame()
for m in models:
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    df = df[(df['session'] > 6)]
    df = df[(df['predict'] == 1)]
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    # df = df[(df['group'] != 'iggJos')]

    df['correct'] = df['animal'] == df['result']

    T = df.groupby(['animal', 'hour'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'hour'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = m

    T = T.sort_values(by=['animal', 'hour'])
    T = T.reset_index(drop=True)

    DATA_stat = DATA_stat.append(T)

DATA_stat = DATA_stat.reset_index(drop=True)

sns.boxenplot(x="Model", y="accuracy", hue="Model", dodge=False, data=DATA_stat, palette=palette, ax=ax[1])
# g = sns.stripplot(x="Model", y="accuracy", hue="Model", dodge=True, color='0.3',
#                   size=4, linewidth=0, data=DATA_stat, ax=ax[1])
# ax[1].set_ylim([0.75, 1.01])

# t, p = ranksums(DATA_stat[DATA_stat['Model'] == models[0].split('_')[1]]['accuracy'],
#                 DATA_stat[DATA_stat['Model'] == models[1].split('_')[1]]['accuracy'])
#
# if p > 0.05:
#     ax[0, 1].text(0.5, 0.8, 'n.s.', color='black', fontsize=12, va="bottom", ha="center")
# else:
#     ax[0, 1].text(0.5, 0.8,'p = ' "%.4f" % p, color='black', fontsize=12, va="bottom", ha="center")

ax[1].set_xlabel(xlabel='Models')
g.set(xticklabels=[])
ax[1].legend([], [], frameon=False)
ax[1].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)

# plot_name = 'mID_AllNets'
# NAME = "{}{}{}".format(PLOT_path, plot_name, '.pdf')
# plt.savefig(NAME, format='pdf')
# NAME = "{}{}{}".format(PLOT_path, plot_name, '.png')
# plt.savefig(NAME, format='png')
