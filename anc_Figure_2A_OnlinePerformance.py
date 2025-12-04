from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dPrime_2AFC as dPrime
from scipy.io import wavfile
from scipy.stats import ranksums
from scipy.stats import ttest_1samp
from scipy import stats
from scipy.stats import binom_test
import os
import glob

# Panel A - All Networks performance
# =============================================
# Setting plotting parameters
plot_name = 'Figure_2A'

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

df['correct'] = df['animal'] == df['result']
categories = df['animal'].unique()
groups = df['group'].unique()

T = df.groupby(['group', 'session'])['correct'].sum().reset_index()
T['trials'] = df.groupby(['group', 'session'])['pic_name'].count().reset_index()['pic_name']
T['accuracy'] = T['correct'] / T['trials']
T['wrong'] = T['trials'] - T['correct']
T['trials_log'] = T['trials'].apply(np.log10)

T = T.sort_values(by=['group', 'session'])
T = T.reset_index(drop=True)

# ===== DO THIS IN R
MODELS_df = pd.DataFrame(columns=['group', 'session', 'dprime', 'binomial'], dtype=float)
for g in groups:
    for s in df[df['group'] == g]['session'].unique():
        temp = df[(df['session'] == s) & (df['group'] == g) & (df['predict'] == 1)]
        temp = temp.reset_index(drop=True)
        temp['correct'] = temp['animal'] == temp['result']
        Pc = sum(temp['correct']) / len(temp)

        # a = df[(df['group'] == g) & (df['session'] == s)]['animal'].unique()[0]
        # b = df[(df['group'] == g) & (df['session'] == s)]['animal'].unique()[1]
        #
        # A = [len(df[(df['animal'] == a) & (df['session'] == s) & (df['group'] == g) & (df['result'] == a)]),
        #      len(df[(df['animal'] == a) & (df['session'] == s) & (df['group'] == g) & (df['result'] == b)])]
        #
        # B = [len(df[(df['animal'] == b) & (df['session'] == s) & (df['group'] == g) & (df['result'] == a)]),
        #      len(df[(df['animal'] == b) & (df['session'] == s) & (df['group'] == g) & (df['result'] == b)])]
        #
        # rsp = np.array([A, B])
        # dPr = dPrime.dPrime_2AFC(rsp[0, :], rsp[1, :])

# ========== PLOT
figure_height = (90 / 25.4) * sizeMult
figure_width = (120 / 25.4) * sizeMult

fig, ax = plt.subplots(2, 4, constrained_layout=True, sharex='row', sharey='row',
                       figsize=(figure_width, figure_height))
ax = ax.flatten()

groups = T['group'].unique()
group_names = ['Group A', 'Group B', 'Group C', 'Group D']
#palette = {groups[0]: "orange", groups[1]: "blue", groups[2]: "purple", groups[3]: "green"}
cols = sns.color_palette("tab10", n_colors=4)
palette = dict(zip(groups, cols))

for idx, g in enumerate(groups):
    sns.barplot(x="session", y="trials_log", data=T[T['group'] == g], color=palette[g], ci=95, ax=ax[idx])
    ax[idx].set_title(group_names[idx], fontsize=labelFontSize)
    #ax[idx].set_yscale('log')
    ax[idx].set_ylabel(ylabel=None)
    ax[idx].set_xlabel(xlabel=None)
    ax[idx].legend([], [], frameon=False)
    #ax[idx].yaxis.set_major_locator(plt.MaxNLocator(3))

ax[0].set_yticklabels([0, 10, 100, 1000])
ax[0].set_yticks([0, 1, 2, 3])

for idx, g in enumerate(groups):
    sns.barplot(x="session", y="accuracy", data=T[(T['group'] == g) & (T['session'] > 3)],
                color=palette[g], ci=95, ax=ax[idx+4])
    ax[idx+4].set_ylabel(ylabel=None)
    ax[idx+4].set_xlabel(xlabel=None)
    ax[idx+4].legend([], [], frameon=False)

    med = np.mean(T[(T['group'] == g) & (T['session'] > 3)]['accuracy'])
    ax[idx+4].hlines(y=med, xmin=-0.5, xmax=4.5, linewidth=1.5, color='k', linestyles='dashed')

ax[0].set_ylim([1,4])
ax[4].set_ylim([0.8, 1.01])
ax[4].set_yticks([0.8, 0.85, 0.9, 0.95, 1])

ax[4].set_xlabel(xlabel="Sessions")
ax[4].set_ylabel(ylabel="Accuracy")
ax[0].set_ylabel(ylabel="Trials")

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
