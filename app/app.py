import streamlit as st
import os
from zipfile import ZipFile
import zipfile
import cv2
import numpy as np

#Load the model and save it on the "cache" memory if we dont have the API yet
def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
st.subheader("Image")

def load_images_from_file():
    images = []
    for filename in os.listdir():
        img = cv2.imread(os.path.join(tempDir,filename))
        if img is not None:
            images.append(img)
    return images

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


image_types = ['jpg','png','jpeg']
# image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
zip_file = st.file_uploader("Upload Zip", type=["zip"])

X_pred = None

if zip_file is not None:

    all_imgs = []
    with zipfile.ZipFile(zip_file) as z:
        #Select only images
        for file in z.namelist():
            if file.split('.')[-1] in image_types:
                #Extract image locally
                try:
                    extracted_file = z.extract(file)
                    img = cv2.imread(extracted_file)
                    img = cv2.resize(img,(32,32))
                    all_imgs.append(img)
                #Maybe here is a good moment to predict (#Predict the images) connect to api end point
                #1) load the image variable
                #2) Do the prediction
                #3) Save the prediction
                #4) Create a new file name
                #5) Append the new file to a list
                    print("Extracted all data")
                except Exception as e:
                    print("Invalid file ", str(e))
    X_pred = np.stack(all_imgs , axis = 0)
    #Create the final zip file
    #Make a botton available that the user can download the zip file back
    #Summary for the prediction in each file and pass what is the probability that our model is outputing

#User can download the finalzip
# st.download_button(
#         label="Download ZIP",
#         data=zip_file,
#         file_name="myfile.zip",
#         mime="application/zip"
#     )
# with open("myfile.zip", "rb") as fp:
#     btn = st.download_button(
#         label="Download ZIP",
#         data=fp,
#         file_name="myfile.zip",
#         mime="application/zip"
#     )
