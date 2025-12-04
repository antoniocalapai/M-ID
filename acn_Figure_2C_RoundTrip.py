from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.stats import ranksums
from scipy.stats import ttest_1samp
from scipy.stats import norm
import os
import glob

# Panel A - All Networks performance
# =============================================
# Setting plotting parameters
plot_name = 'Figure_2C'

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

df = df[(df['predict'] == 1)]

figure_height = (45 / 25.4) * sizeMult
figure_width = (45 / 25.4) * sizeMult

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(figure_width, figure_height))
sns.histplot(df['round_trip']/1000, bins=20, ax=ax, color='black')

m = np.mean(df['round_trip']/1000)
ci = norm(*norm.fit(df['round_trip']/1000)).interval(0.95)
ax.axvline(x=m, linewidth=1.5, color='red',linestyle='--')
ax.text(np.mean(df['round_trip']/1000)-3, 10000, "%.0f"%m, ha='right', va='bottom', color='r', fontsize=12)
ax.set_ylabel(ylabel=None)
ax.set_xlabel(xlabel='Round Trip [ms]')
ax.set_xlim([100, 250])
ax.set_xticks([100, 150, 200, 250])
ax.set_yticklabels([])

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
