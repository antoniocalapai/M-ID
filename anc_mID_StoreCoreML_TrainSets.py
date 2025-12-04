from progressbar import progressbar
import numpy as np
import pandas as pd
import pathlib
import glob
import os

IMG_path_Alw = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/AlwCla/train/Alw/'
IMG_path_Cla = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/AlwCla/train/Cla/'
IMG_path_Der = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/DerElm/train/Der/'
IMG_path_Elm = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/DerElm/train/Elm/'
IMG_path_Edg = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/EdgSun/train/Edg/'
IMG_path_Sun = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/EdgSun/train/Sun/'
IMG_path_Igg = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/IggJos/train/Igg/'
IMG_path_Jos = '/Volumes/DPZ/KognitiveNeurowissenschaften/WeCo/MonkeyID/ML_models/used_pictures/IggJos/train/Jos/'

DF_path = './dataframes/'

csv_file = "{}{}".format(DF_path, 'BaseDataFrame.csv')
df = pd.read_csv(csv_file, index_col=0, low_memory=False)
df['train'] = 0

# ==========================================================================================
T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Alw, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Alw: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'alw']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Cla, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Cla: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'cla']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Der, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Der: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'der']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Elm, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Elm: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'elm']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Edg, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Edg: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'edg']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Sun, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Sun: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'sun']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Igg, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Igg: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'igg']['train'])))

T = pd.DataFrame()
T['fileNames'] = [os.path.basename(x) for x in glob.glob('{}{}'.format(IMG_path_Jos, '*.jpg'))]
T['pic_name'] = T['fileNames'].str[:-10]
for i in progressbar(T['pic_name']):
    if len(df[df['pic_name'] == i]) > 0:
        df.loc[df['pic_name'] == i, 'train'] = 1
print('Jos: ' + str(len(T)) + ' ' + str(sum(df[df['animal'] == 'jos']['train'])))
# ==========================================================================================
df.to_csv("{}{}".format(DF_path, 'CoreML_TrainingSet.csv', sep=','))
