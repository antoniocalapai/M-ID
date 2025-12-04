import time
import cv2
import numpy as np
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

start_time = time.time()
camera_port = 0
IMG_SIZE = 300
camera = cv2.VideoCapture(camera_port)
return_value, image = camera.read()
del camera
print("--- Taking pictures: %s seconds ---" % (time.time() - start_time))
take = (time.time() - start_time)

# ========== Tansform acquired picture into numpy array
start_time = time.time()
image = image / 255.0
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("--- Transform the picture %s seconds ---" % (time.time() - start_time))
transform = (time.time() - start_time)

# ========== Load Model
start_time = time.time()
model = keras.models.load_model('RM_IvJ.model')
print("--- Load the model %s seconds ---" % (time.time() - start_time))
loadModel = (time.time() - start_time)

# ========== Test Picture against the model
start_time = time.time()
result = model.predict([image])
categories = ['igg', 'jos']
result = categories[np.argmax(result)]
print("--- Get Picture ID from model %s seconds ---" % (time.time() - start_time))
getID = (time.time() - start_time)

print('Time w/o model loading [sec]', take + transform + getID)
print('Time w/ model loading [sec]', take + transform + loadModel + getID)

