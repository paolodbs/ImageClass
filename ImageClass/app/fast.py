from fastapi import FastAPI , UploadFile , File , Request
import requests
# ATTENTION not sure of that:
from ImageClass.architecture import initialize_model_3
# ATTENTION not sure of that:
from ImageClass.preprocess import normalizing
from ImageClass.get_data import create_class_dic
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io
from typing import List , Dict
import os
from pathlib import Path
import matplotlib.pyplot as plt

app = FastAPI()

#ATTENTION not very sure of that:
#app.state.model = load_model('../../notebooks/my_model')

##store model to cache mem
app.state.model = load_model('../../raw_data/my_lovely_model')

##store categories to cache mem
app.state.labels = create_class_dic()

@app.get("/")
def home():
    return {"Message": "Welcome to our Image Predictor!"}

#as a reminder now we can run this code in our command line: uvicorn simple:app --reload

#ATTENTION NOT FINISHED!!!!!!!!:

#response = requests.get(url, params=params)
#response.json()

#ATTENTION NOT FINISHED AT ALL!!!!!! and the dataframe type hint is not ok:
#@app.get("/predict")
#def predict(image: dataframe):
#    X_pred = pd.DataFrame(image)
#    model = app.state.model
#    X_processed = normalizing(X_pred)
#    y_pred = model.predict(X_processed)
#    categories = create_class_dic()
#    category = categories.get(y_pred)
#    return {"Here you have the prediction": category}

@app.post('/predict')
async def make_prediction(image : UploadFile = File(...)):
    """Returns the prediction of a given image:
    /TESTED against jpgs
    """
    ##!after plently of debugging a clean solution!

    ##returns the bytes encoding of the image
    request_object_content = await image.read()

    ##creates a PIL object and converts it to RGB
    ##bear in mind PIL cannot directly read the byte string
    ##Hence, we need to save it to the buffer and convert it to RGB (RGB may be by default)
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")

    ## PIL stores images as 32-bits floats by default / No need to resize
    ## - however with a different model architecture
    ## the reshape below might be required
    img_reshaped = cv2.resize(np.array(img) , (32,32))

    ##may need to normalize for better predictions
    ##TODO : uncomment me if u want
    ##img_reshaped = img_reshaped/255

    ##againt np.array([ . ]) expands the dimension by 1 in front ~ required by tensorflow
    prediction =  app.state.model.predict( np.array([img_reshaped]) )

    ##finally predict the category
    ##prediction returns the array of predicted probabilities belonging to each particular class
    ##i.e. model doesn't predict the class but the probabilities of belonging to each particular class
    ##Finally, we choose the class corresponding to the highest predicted probability
    ##and we return the category corresponding to the index with the highest probability
    predicted_category = app.state.labels.get( np.argmax(prediction[0]) )

    ##sending the server response / if status_code == 200
    return {'prediction' : predicted_category}


def preprocess_img(object_content):
    """Basic image preprocessing"""
    img = Image.open(io.BytesIO(object_content)).convert("RGB")
    img_reshaped = cv2.resize(np.array(img) , (32,32))

    #Be wary normalization is wrong; its already normalizing
    #img_reshaped= img_reshaped/255
    return img_reshaped


# @app.post('/multipredict')
# async def get_multiple_predictions(files : List[UploadFile] = File(...)):
#     """Receives a collection of images and returns the dictionary
#     of predicted categories for each image"""
#     content = { item.filename : await item.read()  for item in files }
#     predictions = dict()
#     for name , img in content.items():
#         prediction =  app.state.model.predict( np.array([preprocess_img(img) ]) )
#         predicted_category = app.state.labels.get( np.argmax(prediction[0]) )
#         predictions.update( {name : predicted_category } )
#     return {"predictions": predictions }

api_path = "/home/nicole/code/NicoleChant/ImageClass/ImageClass/app/"

@app.post('/multipredict')
async def get_multiple_predictions(files : List[UploadFile] = File(...)):
    """Receives a collection of images and returns the dictionary
    of predicted categories for each image"""
    content = { item.filename : preprocess_img(await item.read())  for item in files }
    predictions = dict()
    if not os.path.isdir(api_path + "images"):
        os.mkdir(api_path + "images")

    for name , img in content.items():
        prediction =  app.state.model.predict( np.array([img ]) )
        predicted_category = app.state.labels.get( np.argmax(prediction[0]) )
        predictions.update( {name : predicted_category } )
        plt.imshow(img)
        if not os.path.isdir(api_path + f"images/{predicted_category}"):
            os.mkdir(api_path + f"images/{predicted_category}")
        plt.savefig(api_path + f"images/{predicted_category}/{name}")
    return {"predictions": predictions }

@app.post('/filename')
async def pred(image : UploadFile = File(...)):
    """TEST endpoint
                Returns image filename
    /TESTED against pngs"""
    request_object_content = await image.read()
    img = Image.open(io.BytesIO(request_object_content))
    return {"prediction" : image.filename}
