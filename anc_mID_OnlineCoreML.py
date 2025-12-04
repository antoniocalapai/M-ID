from tensorflow import keras
import tensorflow as tf
import numpy as np
import gc
from progressbar import progressbar
import seaborn as sns
import os
from os import listdir
import os.path as up
from datetime import date
import cv2
import random
import pandas as pd
from subprocess import run
import glob
import pathlib

model_name = 'Online_CoreML'
Nsessions = 3
makedf = 1

output = run("pwd", capture_output=True).stdout
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MODEL_path = './CNN_models/'
DF_path = './dataframes/'

# ==========================================================================================
# Load Base Data Frame
csv_file = "{}{}".format(DF_path, 'BaseDataFrame.csv')
df = pd.read_csv(csv_file, index_col=0, low_memory=False)
categories = df['animal'].unique()

df['train'] = 0
df['result'] = 0
df['predict'] = 0
df['CNN_Npics'] = 0
df['CNN_modelBase'] = model_name
df['CNN_Nsessions'] = Nsessions
df['round_trip'] = 0

df = df.reset_index(drop=True)

to_predict = df['session'].unique()[Nsessions:]
for i in to_predict:
    df.loc[df.session == i, 'predict'] = 1

# ==========================================================================================
# Load Experimental Data
csv_file2 = "{}{}".format(DF_path, 'mID_stage1&2_CuratedDataFrame_20220113.csv')
df2 = pd.read_csv(csv_file2, low_memory=False)

notFound = 0
for i in progressbar(df[df['predict'] == 1]['pic_name']):
    if len(df2.loc[df2['picID'] == i, 'ML_label']) > 0:
        df.loc[df['pic_name'] == i, 'result'] = df2.loc[df2['picID'] == i, 'ML_label'].values
        df.loc[df['pic_name'] == i, 'round_trip'] = df2.loc[df2['picID'] == i, 'roundTrip'].values
    else:
        notFound = notFound + 1
print(notFound)

# =====
# Load training set information
csv_file3 = "{}{}".format(DF_path, 'CoreML_TrainingSet.csv')
df3 = pd.read_csv(csv_file3, index_col=0, low_memory=False)

notFound = 0
for i in progressbar(df3[df3['train'] == 1]['pic_name']):
    if len(df3.loc[df3['pic_name'] == i, 'animal']) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
    else:
        notFound = notFound + 1
print(notFound)

save_name = "{}{}{}".format(DF_path, model_name, '.csv')
df.to_csv(save_name, sep=',')
