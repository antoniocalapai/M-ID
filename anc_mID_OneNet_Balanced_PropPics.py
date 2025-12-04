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

model_name = 'OneNet_Balanced_PropPics'
Nsessions = 3
makemodel = 0
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

proportions = list(range(10, 101, 10))
proportions.append(1)
proportions = sorted(proportions)


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

    for i in proportions:
        df['train_' + str(i)] = 0
        df['result_' + str(i)] = 0
        df['CNN_Npics_' + str(i)] = 0

    df['predict'] = 0
    df['CNN_modelBase'] = model_name
    df['CNN_Nsessions'] = Nsessions
    df = df.reset_index(drop=True)

    # ==========================================================================================
    # Use data from the first three sessions to train the model
    to_predict = df['session'].unique()[Nsessions:]
    for i in to_predict:
        df.loc[df.session == i, 'predict'] = 1

    # ==========================================================================================
    # Select training pictures
    for i in progressbar(proportions):
        n_samples = min(df[(df['predict'] == 0)]['animal'].value_counts())
        n_samples = int(i / 100 * n_samples)
        df['CNN_Npics' + str(i)] = np.floor(n_samples)

        for m in categories:
            non_test_list = df[(df['animal'] == m) & (df['predict'] == 0)].pic_name.to_list()
            if len(non_test_list):
                selected = random.sample(non_test_list, n_samples)
                for r in range(0, len(df)):
                    if df.loc[r, 'pic_name'] in selected:
                        df.loc[r, 'train_' + str(i)] = 1

    save_name = "{}{}{}".format(DF_path, model_name, '.csv')
    df.to_csv(save_name, sep=',')

else:
    print('Loading existing dataframe')
    save_name = "{}{}{}".format(DF_path, model_name, '.csv')
    df = pd.read_csv(save_name, index_col=0, low_memory=False)
    categories = df['animal'].unique()

# ==========================================================================================
# Prepare the images for training the network
if makemodel:
    for i in progressbar(proportions):
        print('Training new model')
        X_train = []
        y_train = []

        for m in categories:
            class_num = list(categories).index(m)
            all_elements = df[(df['animal'] == m) & (df['train_' + str(i)] == 1)].pic_name.to_list()

            for img in all_elements:
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
                  epochs=10,
                  batch_size=32)

        NAME = "{}{}{}{}{}".format(MODEL_path, model_name, '_', i, '.model')
        model.save(NAME)

        tf.keras.backend.clear_session()
        gc.collect()
        del model

# ======= TEST THE NETWORK ======
print('Loading existing models')
model_1 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(1), '.model'))
model_10 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(10), '.model'))
model_20 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(20), '.model'))
model_30 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(30), '.model'))
model_40 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(40), '.model'))
model_50 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(50), '.model'))
model_60 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(60), '.model'))
model_70 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(70), '.model'))
model_80 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(80), '.model'))
model_90 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(90), '.model'))
model_100 = keras.models.load_model("{}{}{}{}{}".format(MODEL_path, model_name, '_', str(100), '.model'))

to_predict = df['session'].unique()[Nsessions:]
for i in to_predict:
    df.loc[df.session == i, 'predict'] = 1

for i in progressbar(df[df['predict'] == 1]['pic_name']):

    actual_img_name = df[df['pic_name'] == i]['fileName'].values[0]
    IMG = prepare(os.path.join(IMG_path, actual_img_name))

    result = model_1([IMG])
    df.loc[df['pic_name'] == i, 'result_1'] = categories[np.argmax(result)]

    result = model_10([IMG])
    df.loc[df['pic_name'] == i, 'result_10'] = categories[np.argmax(result)]

    result = model_20([IMG])
    df.loc[df['pic_name'] == i, 'result_20'] = categories[np.argmax(result)]

    result = model_30([IMG])
    df.loc[df['pic_name'] == i, 'result_30'] = categories[np.argmax(result)]

    result = model_40([IMG])
    df.loc[df['pic_name'] == i, 'result_40'] = categories[np.argmax(result)]

    result = model_50([IMG])
    df.loc[df['pic_name'] == i, 'result_50'] = categories[np.argmax(result)]

    result = model_60([IMG])
    df.loc[df['pic_name'] == i, 'result_60'] = categories[np.argmax(result)]

    result = model_70([IMG])
    df.loc[df['pic_name'] == i, 'result_70'] = categories[np.argmax(result)]

    result = model_80([IMG])
    df.loc[df['pic_name'] == i, 'result_80'] = categories[np.argmax(result)]

    result = model_90([IMG])
    df.loc[df['pic_name'] == i, 'result_90'] = categories[np.argmax(result)]

    result = model_100([IMG])
    df.loc[df['pic_name'] == i, 'result_100'] = categories[np.argmax(result)]

save_name = "{}{}{}".format(DF_path, model_name, '.csv')
df.to_csv(save_name, sep=',')
