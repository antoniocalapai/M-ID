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

model_name = 'MultiNet_UnBal_ExtSetSize'
Nsessions = 6
makemodel = 1
makedf = 1

# settings and paths
IMG_SIZE = 300

# Model's parameters (based on best model on Colab)
AvPool = 6
Conv_1 = 64
Conv_2 = 16
Conv_3 = 32
DenLay = 1024

output = run("pwd", capture_output=True).stdout
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


IMG_path = './data/'
MODEL_path = './CNN_models/'
DATA_path = './Pic_Labels/'
DF_path = './dataframes/'


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return img_array


if makedf:
    # ==========================================================================================
    # Load Base Data Frame
    csv_file = "{}{}".format(DF_path, 'BaseDataFrame.csv')
    df = pd.read_csv(csv_file, index_col=0, low_memory=False)

    df['predict'] = 0
    df['train'] = 0
    df['result'] = 0
    df['CNN_Npics'] = 0
    df['CNN_Nsessions'] = Nsessions

    df = df.reset_index(drop=True)

    # ==========================================================================================
    # Use data from the first three sessions to train the model
    to_predict = df['session'].unique()[Nsessions:]
    for i in to_predict:
        df.loc[df.session == i, 'predict'] = 1

    groups = df['group'].unique()
    # ==========================================================================================
    # Select training pictures
    for group in groups:
        animals = df[(df['predict'] == 0) & (df['group'] == group)]['animal'].unique()
        for m in animals:
            df.loc[(df['predict'] == 0) & (df['animal'] == m), 'train'] = 1
            n_samples = sum(df[(df['predict'] == 0) & (df['animal'] == m)]['train'])
            df.loc[df['animal'] == m, 'CNN_Npics'] = n_samples
            df.loc[df['animal'] == group, 'CNN_name'] = "{}{}{}".format(model_name, '_', group)

            non_test_list = df[(df['animal'] == m) & (df['predict'] == 0) & (df['group'] == group)] \
                .pic_name.to_list()
            selected = random.sample(non_test_list, n_samples)
            for r in progressbar(range(0, len(df))):
                if df.loc[r, 'pic_name'] in selected:
                    df.loc[r, 'train'] = 1

    save_name = "{}{}{}".format(DF_path, model_name, '.csv')
    df.to_csv(save_name, sep=',')

else:
    print('Loading existing dataframe')
    save_name = "{}{}{}".format(DF_path, model_name, '.csv')
    df = pd.read_csv(save_name, index_col=0, low_memory=False)
    groups = df['group'].unique()

# ==========================================================================================
# Prepare the images for training the network
if makemodel:
    for group in groups:
        X_train = []
        y_train = []

        animals = df[df['group'] == group]['animal'].unique()
        for m in animals:
            class_num = list(animals).index(m)
            all_elements = df[(df['animal'] == m) & (df['train'] == 1) & (df['group'] == group)]\
                .pic_name.to_list()

            for img in progressbar(all_elements):
                if not img.startswith('.'):
                    if len(df[df['pic_name'] == img]):
                        actual_img_name = df[df['pic_name'] == img]['fileName'].values[0]
                        new_array = prepare(os.path.join(IMG_path, actual_img_name))
                        X_train.append(new_array)
                        y_train.append(class_num)

        X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        y_train = np.array(y_train)

        # ==========================================================================================
        # Build the Convolutional Neural Network
        model = keras.Sequential([
            # Average across AvPool pixels
            keras.layers.AveragePooling2D(AvPool, 3, input_shape=(IMG_SIZE, IMG_SIZE, 1)),

            # Go through Two hidden convolutional layers
            keras.layers.Conv2D(Conv_1, 3, activation='relu'),
            keras.layers.Conv2D(Conv_2, 3, activation='relu'),
            keras.layers.Conv2D(Conv_3, 3, activation='relu'),

            # Pool across hidden layer 2 neurons
            keras.layers.MaxPool2D(2, 2),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),

            # Go through another hidden Layer
            keras.layers.Dense(DenLay, activation='relu'),

            # Output layer
            keras.layers.Dense(len(animals), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        model.fit(X_train, y_train,
                  epochs=10,
                  batch_size=32)

        NAME = "{}{}{}{}{}".format(MODEL_path, model_name, '_', group, '.model')
        model.save(NAME)

        tf.keras.backend.clear_session()
        gc.collect()

# ======= TEST THE NETWORK ======
for group in groups:
    # clean up traces of previous loaded models
    tf.keras.backend.clear_session()
    gc.collect()

    NAME = "{}{}{}{}{}".format(MODEL_path, model_name, '_', group, '.model')
    model = keras.models.load_model(NAME)
    animals = df[df['group'] == group]['animal'].unique()

    for i in progressbar(df[(df['group'] == group) & (df['predict'] == 1)]['pic_name']):
        actual_img_name = df[df['pic_name'] == i]['fileName'].values[0]
        IMG = prepare(os.path.join(IMG_path, actual_img_name))
        result = model([IMG])
        df.loc[df['pic_name'] == i, 'result'] = animals[np.argmax(result)]

save_name = "{}{}{}".format(DF_path, model_name, '.csv')
df.to_csv(save_name, sep=',')
