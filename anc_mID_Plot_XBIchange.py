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
models = ['MultiNet_SameAsCoreML', 'OneNet_UnBalanced', 'Online_CoreML', 'OneNet_Bal_ExtSetSize']
models_name = ['Ker_MultiNet', 'Ker_OneNet', 'CoreML', 'Ker_MultiNet_ext']
palette = {models_name[0]: "b", models_name[1]: "g", models_name[2]: "orange", models_name[3]: "red"}
DATA = pd.DataFrame()

for idx, m in enumerate(models):
    df_name = "{}{}{}".format(DF_path, m, '.csv')
    df = pd.read_csv(df_name, index_col=0, low_memory=False)
    print(m, len(df[df['train'] == 1]))
    df = df[(df['predict'] == 1)]
    df = df[(df['session'] > 6)]

    df['time'] = pd.to_datetime(df['time'], format='%H%M%S')
    df['hour'] = pd.to_datetime(df['time'].dt.floor('h')).dt.time
    df['hour'] = df['hour'].astype(str).str[:2]
    df['hour'] = df['hour'].apply(pd.to_numeric)

    df['correct'] = df['animal'] == df['result']
    categories = df['animal'].unique()

    T = df.groupby(['animal', 'session', 'group'])['correct'].sum().reset_index()
    T['trials'] = df.groupby(['animal', 'session', 'group'])['pic_name'].count().reset_index()['pic_name']
    T['accuracy'] = T['correct'] / T['trials']
    T['wrong'] = 1 - (T['correct'] / T['trials'])
    T['model'] = models_name[idx]

    T = T.sort_values(by=['session', 'animal', 'model'])
    T = T.reset_index(drop=True)

    DATA = DATA.append(T)

DATA = DATA.reset_index(drop=True)

# PLOT
sns.set(font_scale=1.2)
g = sns.FacetGrid(DATA, col='session', height=4, aspect=1, sharex=False)
g.map(sns.barplot, 'animal', 'accuracy', 'model', palette=palette)
g.set_titles(row_template='{row_name}')
g.add_legend()

plot_name = 'mID_ChangeXBI_iggJos'
NAME = "{}{}{}".format(PLOT_path, plot_name, '.pdf')
plt.savefig(NAME, format='pdf')
NAME = "{}{}{}".format(PLOT_path, plot_name, '.png')
plt.savefig(NAME, format='png')
