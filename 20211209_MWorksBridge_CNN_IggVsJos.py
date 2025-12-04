import time
import cv2
import numpy as np
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

camera_port = 0
IMG_SIZE = 300

categories = ['igg', 'jos']
model = keras.models.load_model('RM_IvJ.model')


def prepare(IMG):
    img_array = IMG / 255.0
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return img_array


camera = cv2.VideoCapture(camera_port)
return_value, IMG = camera.read()
IMG = prepare(IMG)

# ========== Load Model
start_time = time.time()
print("--- Load the model %s seconds ---" % (time.time() - start_time))
loadModel = (time.time() - start_time)

# ========== Test Picture against the model
start_time = time.time()
result = model.predict([IMG])
result = categories[np.argmax(result)]
