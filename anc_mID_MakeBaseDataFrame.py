from progressbar import progressbar
import numpy as np
import pandas as pd
import pathlib
import glob
import os

IMG_path = './data/'
MODEL_path = './CNN_models/'
DATA_path = './Pic_Labels/'
DF_path = './dataframes/'

# ==========================================================================================
# Create dataframe with all pictures
df = pd.DataFrame()
df['fileName'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path, '*.jpg'))]
df['pic_name'] = df['fileName'].str[:-10]
df['animal'] = 0

# Create a dataframe with all the labels
label_df = pd.DataFrame()
excel_files = list(pathlib.Path(DATA_path).glob('*.xlsx'))
for i in excel_files:
    T = pd.read_excel(i)

    if not 'edgSun_20211228' in str(i):
        T.drop('notes', axis=1, inplace=True)
        T.drop('user', axis=1, inplace=True)
        if 'app_version' in T:
            T.drop('app_version', axis=1, inplace=True)

        label_df = label_df.append(T)

label_df['pic_name'] = label_df['filename'].str[:-10]

# Merge the two dataframes
for i in progressbar(df['pic_name']):
    if len(label_df[label_df['pic_name'] == i]['manual_label'].values):
        df.loc[df['pic_name'] == i, 'animal'] = label_df[label_df['pic_name'] == i]['manual_label'].values
    else:
        df.loc[df['pic_name'] == i, 'animal'] = 'NotFound'

# ==========================================================================================
# Clean the dataframe
df = df[df['animal'] != 'NotFound']
df = df.dropna()

df.replace('Alv', 'Alw', inplace=True)  # fix labeling mistake with Alwin
df['animal'] = df['animal'].str.lower()  # lowercase for all labels

df['group'] = df['fileName'].str.split('_').apply(lambda x: x[0])  # extract group
df['date'] = df['fileName'].str.split('_').apply(lambda x: x[1])  # extract date
df['time'] = df['fileName'].str.split('_').apply(lambda x: x[2])  # extract time information

df = df.sort_values(by=['group', 'date'])
df = df.reset_index(drop=True)

# ==========================================================================================
# Compute session number across all animals
df['session'] = 0
categories = df['animal'].unique()
for m in categories:
    temp = df[df['animal'] == m]
    temp = temp.sort_values(by=['date'])
    unique_date = temp.date.unique()

    for s in range(0, len(unique_date)):
        df.loc[(df['animal'] == m) & (df['date'] == unique_date[s]), 'session'] = s + 1

df = df.sort_values(by=['group', 'session'])
df = df.reset_index(drop=True)

# ==========================================================================================
# Save dataframe as BaseDataFrame.csv
df.to_csv("{}{}".format(DF_path, 'BaseDataFrame.csv', sep=','))
