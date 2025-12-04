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
# TODO Compare MultiNet Balanced vs Unbalanced
figure_height = (150 / 25.4) * sizeMult
figure_width = (180 / 25.4) * sizeMult

fig, ax = plt.subplots(3, 2, constrained_layout=True, sharex=False, sharey=False,
                       figsize=(figure_width, figure_height), gridspec_kw={'width_ratios': [1, 0.3]})

models = ['MultiNet_Balanced', 'MultiNet_UnBalanced']
MultiNet = pd.DataFrame()
for m in models:
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    print(m, len(df[df['train'] == 1]))
    df = df[(df['predict'] == 1)]
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df = df[(df['group'] != 'iggJos')]

    df['correct'] = df['animal'] == df['result']
    categories = df['animal'].unique()

    T = df.groupby(['animal', 'session', 'hour'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'session', 'hour'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = m.split('_')[1]

    T = T.sort_values(by=['session', 'animal', 'hour'])
    T = T.reset_index(drop=True)

    MultiNet = MultiNet.append(T)

MultiNet = MultiNet.reset_index(drop=True)

sns.barplot(x="session", y="accuracy", hue="Model", data=MultiNet, ci=95, ax=ax[0, 0])
sns.barplot(x="animal", y="accuracy", hue="Model", data=MultiNet, ci=95, ax=ax[1, 0])
sns.barplot(x="hour", y="accuracy", hue="Model", data=MultiNet, ci=95, ax=ax[2, 0])

ax[0, 0].set_ylim([0.75, 1])
ax[1, 0].set_ylim([0.75, 1])
ax[2, 0].set_ylim([0.75, 1])

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

# ==
MultiNet_stat = pd.DataFrame()
for m in models:
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    df = df[(df['predict'] == 1)]
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df = df[(df['group'] != 'iggJos')]

    df['correct'] = df['animal'] == df['result']

    T = df.groupby(['animal', 'hour'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'hour'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = m.split('_')[1]

    T = T.sort_values(by=['animal', 'hour'])
    T = T.reset_index(drop=True)

    MultiNet_stat = MultiNet_stat.append(T)

MultiNet_stat = MultiNet_stat.reset_index(drop=True)

sns.boxplot(x="Model", y="accuracy", hue="Model", whis=[0, 100], data=MultiNet_stat, ax=ax[0, 1])
g = sns.stripplot(x="Model", y="accuracy", hue="Model", dodge=True, color='0.3',
                  size=4, linewidth=0, data=MultiNet_stat, ax=ax[0, 1])

t, p = ranksums(MultiNet_stat[MultiNet_stat['Model'] == 'UnBalanced']['accuracy'],
                MultiNet_stat[MultiNet_stat['Model'] == 'Balanced']['accuracy'])

if p > 0.05:
    ax[0, 1].text(0.5, 0.8, 'n.s.', color='black', fontsize=12, va="bottom", ha="center")
else:
    ax[0, 1].text(0.5, 0.8, "%.2f" % p, color='black', fontsize=12, va="bottom", ha="center")

ax[0, 1].legend([], [], frameon=False)
ax[0, 1].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)
ax[0, 1].set_xlabel(xlabel='Models', fontsize=labelFontSize)
ax[0, 1].tick_params(labelsize=tickFontSize)
g.set(xticklabels=[])

# ==
MultiNet_stat = pd.DataFrame()
for idx, m in enumerate(models):
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    df = df[(df['predict'] == 1)]
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df = df[(df['group'] != 'iggJos')]
    df['correct'] = df['animal'] == df['result']

    T = df.groupby(['animal', 'session', 'hour'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'session', 'hour'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = m.split('_')[1]

    T = T.sort_values(by=['animal', 'session', 'hour'])
    T = T.reset_index(drop=True)

    MultiNet_stat[m.split('_')[1]] = T.accuracy.values
    MultiNet_stat['hour'] = T.hour

MultiNet_stat['Balanced-Unbalanced'] = MultiNet_stat[models[0].split('_')[1]] - MultiNet_stat[models[1].split('_')[1]]
sns.histplot(MultiNet_stat['Balanced-Unbalanced'], bins=20, ax=ax[1, 1])
ax[1, 1].set_xlim([-0.2, 0.2])
ax[1, 1].axvline(MultiNet_stat['Balanced-Unbalanced'].mean(), color='red', linestyle='--', alpha=0.7)
t, p = ttest_1samp(MultiNet_stat['Balanced-Unbalanced'], popmean=0)

if p > 0.05:
    ax[1, 1].text(-0.18,140, 'n.s.', color='black', fontsize=12, va="bottom", ha="left")
else:
    ax[1, 1].text(-0.18,140, "%.2f" % p, color='black', fontsize=12, va="bottom", ha="left")

ax[1, 1].set_ylabel(ylabel=None)
ax[1, 1].set_xlabel(xlabel='Accuracy\n[Balanced - Unbalanced]', fontsize=labelFontSize)

# ==
MultiNet_stat = pd.DataFrame()
for m in models:
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    df = df[(df['predict'] == 1)]
    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df = df[(df['group'] != 'iggJos')]
    df['correct'] = df['animal'] == df['result']

    T = df.groupby(['hour', 'session'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['hour', 'session'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = m.split('_')[1]

    T = T.sort_values(by=['hour', 'session'])
    T = T.reset_index(drop=True)

    MultiNet_stat = MultiNet_stat.append(T)

MultiNet_stat = MultiNet_stat.reset_index(drop=True)
sns.regplot(x='hour', y='accuracy', data=MultiNet_stat[MultiNet_stat['Model'] == 'Balanced'], truncate=False, ax=ax[2,1])
sns.regplot(x='hour', y='accuracy', data=MultiNet_stat[MultiNet_stat['Model'] == 'UnBalanced'], truncate=False, ax=ax[2,1])
ax[2, 1].set_ylim([0.80, 1.01])

X = MultiNet_stat[MultiNet_stat['Model'] == 'Balanced']['hour']
Y = MultiNet_stat[MultiNet_stat['Model'] == 'Balanced']['accuracy']
R, p = stats.pearsonr(X, Y)

ax[2, 1].text(8, 0.85, "{}{}{}{}".format('pearson R = ', "%.2f"%R, ' ; p = ', "%.2f"%p),
              color='blue', fontsize=12, va="bottom", ha="left")

X = MultiNet_stat[MultiNet_stat['Model'] == 'UnBalanced']['hour']
Y = MultiNet_stat[MultiNet_stat['Model'] == 'UnBalanced']['accuracy']
R, p = stats.pearsonr(X, Y)

ax[2, 1].text(8, 0.85, "{}{}{}{}".format('pearson R = ', "%.2f"%R, ' ; p = ', "%.2f"%p),
              color='orange', fontsize=12, va="top", ha="left")

ax[2, 1].set_xlabel(xlabel='Time of Day', fontsize=labelFontSize)
ax[2, 1].set_ylabel(ylabel=None)
ax[2, 1].tick_params(labelsize=tickFontSize)


ax[0, 0].legend(prop={'size': 14}, loc='lower left')
ax[1, 0].legend([], [], frameon=False)
ax[2, 0].legend([], [], frameon=False)

plot_name = 'MultiNet_BalancedVsUnbalanced'
NAME = "{}{}{}".format(PLOT_path, plot_name, '.pdf')
plt.savefig(NAME, format='pdf')
NAME = "{}{}{}".format(PLOT_path, plot_name, '.png')
plt.savefig(NAME, format='png')
