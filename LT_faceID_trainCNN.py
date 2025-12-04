from tensorflow import keras
import tensorflow as tf
import numpy as np
import glob
import os
import shutil
import cv2
import random
import pandas as pd
from subprocess import run

# ==============================
# Global parameters and settings
TEST_N = 50
IMG_SIZE = 300
makemodel = 0

# Categories are hard coded to fit the experiment's categories' order
categories = ['N', 'B', 'L', 'K']

# Model's parameters (based on best model on Colab)
AvPool = 6
Conv_1 = 64
Conv_2 = 16
Conv_3 = 32
DenLay = 1024

# ==================
# settings and paths
output = run("pwd", capture_output=True).stdout
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
img_path = '/Users/acalapai/ownCloud/Shared/mXBI_FaceID/longtailed/data'
new_data_path = '/Users/acalapai/ownCloud/Shared/mXBI_FaceID/longtailed/_newSessions/'

# =================
# custom functions
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return img_array


if makemodel:
    # =================
    # Check for new sessions in _newSessions and move all files:
    if len(os.listdir(new_data_path)) > 1:
        all_new_subfolders = os.listdir(new_data_path)
        for i in all_new_subfolders:
            if i.startswith(".") == 0:
                temp_path = "{}{}".format(new_data_path, i)
                csv_temp_file = glob.glob(temp_path + "/*.csv")
                temp_df = pd.read_csv(csv_temp_file[0], low_memory=False, sep=',', usecols=['FileName', 'animalID'])

                # Move the CSV file to the DATA folder
                new_destination = "{}{}{}".format(img_path, '/', csv_temp_file[0].split('/')[-1])
                shutil.move(csv_temp_file[0], new_destination)

                # List all images in the folder
                all_images = glob.glob(temp_path + "/*.jpg")
                for j in all_images:
                    target = "{}{}{}".format(img_path, '/', j.split('/')[-1])
                    shutil.move(j, target)

                os.rmdir(temp_path)

    # =================
    # Concatenate all csv files in the data folder
    df = pd.DataFrame(columns=['FileName', 'animalID'])
    all_csv_files = glob.glob(img_path + "/*.csv")

    for i in all_csv_files:
        reader = pd.read_csv(i, sep=None, iterator=True)
        inferred_sep = reader._engine.data.dialect.delimiter

        temp_df = pd.read_csv(i, low_memory=False, sep=inferred_sep, usecols=['FileName', 'animalID'])
        df = pd.concat([df, temp_df])

    #df = df[['FileName', 'animalID']]
    df = df.reset_index(drop=True)
    #df.dropna(subset=['FileName'], how='all', inplace=True)
    df["animalID"] = df['animalID'].replace(np.nan, 'N', regex=True)
    df['animalID'] = df['animalID'].str[0:1]
#    df = df.replace(to_replace=r'\\', value='', regex=True)

    # ==================================
    # Add columns for CNN test and train
    df['test'] = 0
    df['train'] = 0
    df['result'] = 0
    df['confidence'] = 0

    # ======================================================
    # Select 50 random pictures for each animal as test set
    for m in df.animalID.unique():
        selected = random.sample(df[df['animalID'] == m].FileName.to_list(), TEST_N)
        for r in range(0, len(df)):
            if df.iloc[r, 0] in selected:
                df.iloc[r, 2] = 1

    # ====================================
    # Select pictures for the training set (excluding N)
    n_samples = min(df[(df['test'] == 0) & (df['animalID'] != 'N')]['animalID'].value_counts() - TEST_N)
    unique_animals = df.animalID.unique()
    for m in unique_animals:
        if m != 'N':
            n_samples = min(df[(df['test'] == 0) & (df['animalID'] != 'N')]['animalID'].value_counts() - TEST_N)
        else:
            n_samples = len(df[df.animalID == 'N']) - TEST_N
        non_test_list = df[(df['animalID'] == m) & (df['test'] == 0)].FileName.to_list()
        selected = random.sample(non_test_list, n_samples)
        for r in range(0, len(df)):
            if df.iloc[r, 0] in selected:
                df.iloc[r, 3] = 1

    # ==================
    # Save the dataframe
    df.to_csv('/Users/acalapai/ownCloud/Shared/mXBI_FaceID/longtailed/img_DF.csv', sep=',')

    # ===========================================
    # Prepare the images for training the network
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    counter = 0

    # =================
    # Process all pictures
    for category in categories:
        class_num = categories.index(category)

        all_elements = df[(df['animalID'] == category) & (df['train'] == 1)].FileName.to_list()
        for img in all_elements:
            new_array = prepare(os.path.join(img_path, img))
            X_train.append(new_array)
            y_train.append(class_num)
            counter = counter + 1
            print(len(all_elements) - counter)

        all_elements = df[(df['animalID'] == category) & (df['test'] == 1)].FileName.to_list()
        for img in all_elements:
            new_array = prepare(os.path.join(img_path, img))
            X_test.append(new_array)
            y_test.append(class_num)

    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # =========
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
        keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=10,
              batch_size=32)

    # ==================
    # Evaluate the model
    R = model.evaluate(X_test, y_test)

    path = '/Users/acalapai/ownCloud/Shared/mXBI_FaceID/longtailed/'
    NAME = "{}{}{}".format(path, 'LT', '.model')
    model.save(NAME)

# Load the existing model and image dataframe
else:
    model = keras.models.load_model(
        '/Users/acalapai/ownCloud/Shared/mXBI_FaceID/longtailed/LT.model')

    csv_file = '/Users/acalapai/ownCloud/Shared/mXBI_FaceID/longtailed/img_DF.csv'
    df = pd.read_csv(csv_file, low_memory=False)

# =========================================================================
# Feed test images directly to the CNN one by one and check the performance
test_run = pd.DataFrame(columns=['image', 'label', 'result', 'confidence'])
for category in categories:
    all_elements = df[(df['animalID'] == category) & (df['train'] == 0)].FileName.to_list()

    for img in all_elements:
        IMG = prepare(os.path.join(img_path, img))
        result = model.predict([IMG])

        test_run = test_run.append({
            'image': img,
            'label': category,
            'result': categories[np.argmax(result)],
            'confidence': result},
            ignore_index=True)

failures = test_run[test_run['label'] != test_run['result']]
print((1 - (len(failures) / len(test_run))) * 100)
