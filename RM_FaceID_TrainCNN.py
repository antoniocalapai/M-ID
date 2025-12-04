from tensorflow import keras
import tensorflow as tf
import numpy as np
import seaborn as sns
import os
import cv2
import random
import pandas as pd
from subprocess import run

# ==============================
# Global parameters and settings
TEST_N = 100
IMG_SIZE = 300
makemodel = 0
IvJ_EvS = 0

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

# ========================
# Select which group
group = ['IvJ', 'EvS']
img_path = "{}{}".format('./RM_pics_', group[IvJ_EvS])

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = img_array / 255.0
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return img_array


if makemodel:
    csv_file = "{}{}{}".format('./img_DF_', group[IvJ_EvS],'.csv')

    # ========================================
    # Import csv files and merge them into one
    df = pd.read_csv(csv_file, low_memory=False, index_col=0, sep=',')

    df['date'] = df['FileName'].str[0:8]
    df['test'] = 0
    df['train'] = 0
    df['result'] = 0
    df['confidence'] = 0

    df = df[df['FileName'] != '.DS_Store']
    df = df.sort_values(by=['date'])
    df = df.reset_index(drop=True)

    # ======================================================
    # Select random pictures for each animal as test set
    for m in df.animalID.unique():
        selected = random.sample(df[df['animalID'] == m].FileName.to_list(), TEST_N)
        for r in range(0, len(df)):
            if df.iloc[r, 0] in selected:
                df.iloc[r, 2] = 1

    # ====================================
    # Select pictures for the training set
    n_samples = min(df[df['test'] == 0]['animalID'].value_counts() - TEST_N)
    for m in df.animalID.unique():
        non_test_list = df[(df['animalID'] == m) & (df['test'] == 0)].FileName.to_list()
        selected = random.sample(non_test_list, n_samples)
        for r in range(0, len(df)):
            if df.iloc[r, 0] in selected:
                df.iloc[r, 3] = 1

    # ==================
    # Save the dataframe
    save_name = "{}{}{}".format('./img_DF_', group[IvJ_EvS], '.csv')
    df.to_csv(save_name, sep=',')

    # ===========================================
    # Prepare the images for training the network
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    categories = df.animalID.unique().tolist()
    for category in categories:
        class_num = categories.index(category)

        all_elements = df[(df['animalID'] == category) & (df['train'] == 1)].FileName.to_list()
        for img in all_elements:
            if not img.startswith('.'):
                new_array = prepare(os.path.join(img_path, img))
                X_train.append(new_array)
                y_train.append(class_num)

        all_elements = df[(df['animalID'] == category) & (df['test'] == 1)].FileName.to_list()
        for img in all_elements:
            if not img.startswith('.'):
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
        keras.layers.Dense(2, activation='softmax')
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

    NAME = "{}{}{}".format('RM_', group[IvJ_EvS], '.model')
    model.save(NAME)

# Load the existing model and image dataframe
else:
    NAME = "{}{}{}".format('RM_', group[IvJ_EvS], '.model')
    model = keras.models.load_model(NAME)

    csv_file = "{}{}{}".format('./img_DF_', group[IvJ_EvS], '.csv')
    df = pd.read_csv(csv_file, index_col=0, low_memory=False)

# =========================================================================
# Feed test images directly to the CNN one by one and check the performance
test_run = pd.DataFrame(columns=['image', 'session', 'label', 'result', 'conf_igg', 'conf_jos'])
categories = df.animalID.unique().tolist()
for category in categories:
    all_elements = df[(df['animalID'] == category) & (df['test'] == 1)].FileName.to_list()

    for img in all_elements:
        if not img.startswith('.'):
            IMG = prepare(os.path.join(img_path, img))
            result = model.predict([IMG])

            test_run = test_run.append({
                'image': img,
                'session': img.split('_')[0],
                'label': category,
                'result': categories[np.argmax(result)],
                'conf_igg': result[0,0],
                'conf_jos': result[0,1]},
                ignore_index=True)

test_run.to_csv('./test_run_DF.csv', sep=',')

failures = test_run[test_run['label'] != test_run['result']]
print((1 - (len(failures) / len(test_run))) * 100)


