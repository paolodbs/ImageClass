from fastapi import FastAPI , UploadFile , File
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
app = FastAPI()
#ATTENTION not very sure of that:
app.state.model = load_model('../../notebooks/my_model')
app.state.labels = create_class_dic()

@app.get("/")
def index():
    return {"Welcome to our Image Predictor!": True}

#as a reminder now we can run this code in our command line: uvicorn simple:app --reload

url = 'http://localhost:8000/predict'

#ATTENTION NOT FINISHED!!!!!!!!:
params = {
    'image': "dunno lol"
}

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

def predict(image):
    X_pred = cv2.imread(image.file.read())
    X_pred = cv2.resize(X_pred , (32,32))
    prediction = app.state.model.predict(X_pred)
    return prediction

@app.post('/predict')
def make_prediction(image : UploadFile = File(...)):
    prediction = predict(image)
    predicted_category = app.state.labels.get( np.argmax(prediction[0]) )
    return {'prediction':predicted_category}
