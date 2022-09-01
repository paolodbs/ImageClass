from fastapi import File, FastAPI, UploadFile
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
import cv2

labels = {0: 'airplane',
 1: 'car',
 2: 'cat',
 3: 'dog',
 4: 'flower',
 5: 'fruit',
 6: 'motorbike',
 7: 'person'}

app = FastAPI()

@app.get('/')

def home():
    return {'Welcome':'To my Image Classification task!'}

@app.post('/predict')

def get_image(file : UploadFile = File(...)):
    image = np.array(Image.open(file.file)).astype('float32')/255
    image_resized = cv2.resize(image, (64,64))
    model = load_model("../../first_natural_model")
    #img_processed = cv2.imread(img_bytes)
    y_pred = model.predict(np.array([image_resized]))
    category = np.argmax(y_pred)
    probabilities_per_category = y_pred[0]
    probabilities = {labels.get(idx) : float(probability) for idx , probability in enumerate(probabilities_per_category)}
    return {"prediction" : labels.get(int(category)),
            "probabilities": probabilities}
