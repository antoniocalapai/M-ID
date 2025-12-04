from tensorflow import keras
import tensorflow as tf
import numpy as np
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

model_name = 'OneNet_Balanced'
Nsessions = 3
makemodel = 1
makedf = 0

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
    print('Creating dataframe')
    # ==========================================================================================
    # Load Base Data Frame
    csv_file = "{}{}".format(DF_path, 'BaseDataFrame.csv')
    df = pd.read_csv(csv_file, index_col=0, low_memory=False)
    categories = df['animal'].unique()

    df['predict'] = 0
    df['train'] = 0
    df['result'] = 0
    df['CNN_Npics'] = 0
    df['CNN_Nsessions'] = Nsessions
    df['CNN_type'] = model_name

    df = df.reset_index(drop=True)

    # ==========================================================================================
    # Use data from the first three sessions to train the model
    to_predict = df['session'].unique()[Nsessions:]
    for i in to_predict:
        df.loc[df.session == i, 'predict'] = 1

    # ==========================================================================================
    # Select training pictures
    n_samples = min(df[(df['predict'] == 0)]['animal'].value_counts())
    df['CNN_Npics'] = n_samples

    for m in categories:
        non_test_list = df[(df['animal'] == m) & (df['predict'] == 0)].pic_name.to_list()
        if len(non_test_list):
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
    categories = df['animal'].unique()

if makemodel:
    print('Training new model')
    # ==========================================================================================
    # Prepare the images for training the network
    X_train = []
    y_train = []

    for m in categories:
        class_num = list(categories).index(m)
        all_elements = df[(df['animal'] == m) & (df['train'] == 1)].pic_name.to_list()

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
        keras.layers.Dense(len(categories), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=1,
              batch_size=32)

    NAME = "{}{}{}".format(MODEL_path, model_name, 'TEST.model')
    model.save(NAME)

else:
    print('Loading existing model')
    NAME = "{}{}{}".format(MODEL_path, model_name, '.model')
    model = keras.models.load_model(NAME)

# ======= TEST THE NETWORK ======
for i in progressbar(df[df['predict'] == 1]['pic_name']):
    actual_img_name = df[df['pic_name'] == i]['fileName'].values[0]
    IMG = prepare(os.path.join(IMG_path, actual_img_name))
    result = model.predict([IMG])
    df.loc[df['pic_name'] == i, 'result'] = categories[np.argmax(result)]

save_name = "{}{}{}".format(DF_path, model_name, '.csv')
df.to_csv(save_name, sep=',')
