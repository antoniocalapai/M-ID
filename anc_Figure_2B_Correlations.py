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
plot_name = 'Figure_2B'

sizeMult = 1
saveplot = 1
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

model = 'Online_CoreML'
df_name = "{}{}{}".format(DF_path, model, '.csv')
df = pd.read_csv(df_name, index_col=0, low_memory=False)

# df = df[(df['predict'] == 1)]
# df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
# df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
# df['hour'] = df['hour'].astype(str).str[:2]
# df['hour'] = df['hour'].apply(pd.to_numeric)

# df = df[(df['group'] != 'iggJos')]

df['correct'] = df['animal'] == df['result']
categories = df['animal'].unique()

T = df.groupby(['group', 'session'])['correct'].sum().reset_index()
T['trials'] = df.groupby(['group', 'session'])['pic_name'].count().reset_index()['pic_name']
T['accuracy'] = T['correct'] / T['trials']
T['wrong'] = T['trials'] - T['correct']
T['trials_log'] = T['trials'].apply(np.log10)
# T['session'] = T['session'] - 3

T = T.sort_values(by=['group', 'session'])
T = T.reset_index(drop=True)

figure_height = (90 / 25.4) * sizeMult
figure_width = (45 / 25.4) * sizeMult

fig, ax = plt.subplots(2, 1, constrained_layout=True, sharex=False, sharey=False,
                       figsize=(figure_width, figure_height))
ax = ax.flatten()

sns.regplot(x='session', y='trials_log', data=T, truncate=False, ax=ax[0],
            scatter_kws={'facecolors': 'none', 'edgecolors': 'gray', }, line_kws={"color": "red"})

ax[0].set_ylabel(ylabel=None)
ax[0].set_xlabel(xlabel=None)
ax[0].set_yticklabels([])
ax[0].set_yticks([0, 1, 2, 3])
ax[0].set_xticks([1, 2, 3, 4, 5, 6, 7, 8])

R, p = stats.pearsonr(T['session'], T['trials_log'])

ax[0].text(8, 0.1, "{}{}{}{}".format('R: ', "%.2f"%R, '\np: ', "%.2f"%p),
           color='k', fontsize=10, va="bottom", ha="right",
           bbox=dict(boxstyle="round", ec=(1, 0, 0), fc=(1, 1, 1), alpha=0.8))

sns.regplot(x='session', y='accuracy', data=T[T['session'] > 3], truncate=False, ax=ax[1],
            scatter_kws={'facecolors': 'none', 'edgecolors': 'gray', }, line_kws={"color": "red"})

ax[1].set_ylabel(ylabel=None)
ax[1].set_xlabel(xlabel='Session')
ax[1].set_yticks([0.8, 0.85, 0.9, 0.95, 1])
ax[1].set_xticks([4, 5, 6, 7, 8])
ax[1].set_yticklabels([])

R, p = stats.pearsonr(T[T['session'] > 3]['session'], T[T['session'] > 3]['trials_log'])

ax[1].text(8, 0.81, "{}{}{}{}".format('R: ', "%.2f"%R, '\np: ', "%.2f"%p),
           color='k', fontsize=10, va="bottom", ha="right",
           bbox=dict(boxstyle="round", ec=(1, 0, 0), fc=(1, 1, 1), alpha=0.8))


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
