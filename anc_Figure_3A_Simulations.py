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
plot_name = 'Figure_3A'

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
models = ['MultiNet_SameAsCoreML', 'Online_CoreML', 'MultiNet_UnBalanced', 'OneNet_UnBalanced']
models_name = ['Keras', 'CoreML', 'MultiNet', 'OneNet']
palette = {models_name[0]: "b", models_name[1]: "orange", models_name[2]: "g", models_name[3]: "purple"}
DATA = pd.DataFrame()

for idx, m in enumerate(models):
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    print(m, len(df[df['train'] == 1]))

    df = df[(df['predict'] == 1)]

    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df = df.drop(df[(df['group'] == exclude_group) & (df['session'] >= exclude_sessions)].index)

    df['correct'] = df['animal'] == df['result']
    categories = df['animal'].unique()

    T = df.groupby(['animal', 'session', 'hour'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'session', 'hour'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = T['trials'] - T['correct']
    T['Model'] = models_name[idx]

    T = T.sort_values(by=['session', 'animal', 'hour'])
    T = T.reset_index(drop=True)

    DATA = DATA.append(T)

DATA = DATA.reset_index(drop=True)

figure_height = (60 / 25.4) * sizeMult
figure_width = (135 / 25.4) * sizeMult

fig, ax = plt.subplots(1, 2, constrained_layout=True, sharex=False, sharey=True, figsize=(figure_width, figure_height))

g = sns.barplot(x="Model", y="accuracy", hue="Model", dodge=False, data=DATA, ci=95, palette=palette, ax=ax[0])
g.legend_.set_title(None)
ax[0].set_title('Models comparison', fontsize=labelFontSize)
ax[0].set_ylim([0.75, 1.01])
ax[0].legend(ncol=2)
ax[0].set_ylabel(ylabel='Accuracy', fontsize=labelFontSize)
ax[0].set_xlabel(xlabel='Models', fontsize=labelFontSize)
ax[0].text(0.5, 0.977, "*", ha='center', va='bottom', fontsize=18)
ax[0].hlines(y=0.997, xmin=0, xmax=1, linewidth=1.5, color='k')
ax[0].set_xticks([0.5, 2.5])
ax[0].set_xticklabels(['Balanced', 'Unbalanced'], fontsize=tickFontSize)

# =============================================
proportions = list(range(10, 101, 10))
proportions.append(1)
proportions = sorted(proportions)

csv_file = 'OneNet_Balanced_PropPics'
df_name = "{}{}{}".format(DF_path, csv_file, '.csv')
df = pd.read_csv(df_name, index_col=0, low_memory=False)

df = df[(df['predict'] == 1)]

df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
df['hour'] = df['hour'].astype(str).str[:2]
df['hour'] = df['hour'].apply(pd.to_numeric)

df = df.drop(df[(df['group'] == exclude_group) & (df['session'] >= exclude_sessions)].index)
df = df.reset_index(drop=True)

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

ax[1] = sns.regplot(x="set_size", y="correct", data=DATA, logx=True,
                    scatter_kws={'facecolors': 'none', 'edgecolors': 'gray', 's': DATA['total'] / 200},
                    line_kws={"color": "red"})

ax[1].set_title('Balanced OneNet', fontsize=labelFontSize)
ax[1].set_xlabel(xlabel='Set Size', fontsize=labelFontSize)
ax[1].set_ylabel(ylabel=None)

ax[1].text(900, 0.77, 'O: 13000 \n o: 700',
           bbox=dict(boxstyle="round", ec=(0.4, 0.4, 0.4), fc=(1, 1, 1),), fontsize=10)  # add text

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
