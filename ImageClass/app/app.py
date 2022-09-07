import streamlit as st
import os
from zipfile import ZipFile
import zipfile
import cv2
import numpy as np
import requests
import os
import io
import matplotlib.pyplot as plt
import shutil

import base64

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def get_prediction(session , uploaded_file):
    response = session.post("http://127.0.0.1:8000/predict",
                    files = {'image': open(uploaded_file,'rb') } )
    return response.json()
#    with open(uploaded_file , 'rb') as f:
#        response = requests.post("http://127.0.0.1:8000/bytesprediction_2",
#                             data = {'image_bytes':f.read()} )
#    return response.json()



#def zip_compression_tree(root):
#    buf = io.BytesIO()
#    with zipfile.ZipFile(buf, mode = 'w', compression=zipfile.ZIP_DEFLATED) as z:
#        for root, dirs, files in os.walk(root):
#            for file in files:
#                z.write(os.path.join(root, file))
#            for directory in dirs:
#                z.write(os.path.join(root, directory))

#Load the model and save it on the "cache" memory if we dont have the API yet
def save_uploadedfile(uploadedfile):
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved File:{uploadedfile.name} to tempDir")
st.subheader("Image")
#def load_images_from_file():
# images = []
# for filename in os.listdir():
# img = cv2.imread(os.path.join(tempDir,filename))
# if img is not None:
# images.append(img)
# return images"""
def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

image_types = ['jpg','png','jpeg']
# image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
zip_file = st.file_uploader("Upload Zip", type=["zip"])

X_pred = None
prediction=None


shutil.rmtree('images' , ignore_errors = True)
shutil.rmtree('images.zip' , ignore_errors = True)
os.system('rm images.zip')
os.system('ls')

if not os.path.isdir('images'):
    os.mkdir('images')

if zip_file is not None:
    #all_imgs = []
    predictions = []
    #content_to_get = []
    with zipfile.ZipFile(zip_file) as z:
        #Select only images
        with requests.Session() as session:

            for file in z.namelist():
                if file.split('.')[-1] in image_types:
                    #Extract image locally
                    try:
                        extracted_file = z.extract(file)
                        prediction = get_prediction(session , extracted_file).get('prediction')
                        predictions.append(prediction)

                        #st.write(prediction)


                        if not os.path.isdir(f"images/{prediction}"):
                            os.mkdir(f"images/{prediction}")

                        print(f'moving file {file}')
                        os.system(f'mv {file} images/{prediction}')
                        #plt.savefig(prediction , f"images/{prediction}")

                        #st.write(extracted_file.split('/')[-1])

                        #img = cv2.imread(extracted_file)
                        #img = cv2.resize(img,(32,32))
                        #all_imgs.append(img)
                    #Maybe here is a good moment to predict (#Predict the images) connect to api end point
                    #1) load the image variable
                    #2) Do the prediction
                    #3) Save the prediction
                    #4) Create a new file name
                    #5) Append the new file to a list
                        print("Extracted and moved file")
                        print(f'prediction: {prediction}')

                    except Exception as e:
                        print("Invalid file ", str(e))
    final_output = {}
    for i , prediction in enumerate(predictions):
        file = z.namelist()[i]
        final_output[file] = prediction


    st.markdown(final_output)
    shutil.make_archive('images', 'zip', 'images')

    #Create the final zip file
    #Make a botton available that the user can download the zip file back
    #Summary for the prediction in each file and pass what is the probability that our model is outputing
#User can download the finalzip

    st.markdown(get_binary_file_downloader_html('images.zip') , unsafe_allow_html=True)
    #st.download_button(
    #        label="Download ZIP",
    #        data='images.zip',
    #        mime="application/zip"
    #    )

# with open("myfile.zip", "rb") as fp:
#     btn = st.download_button(
#         label="Download ZIP",
#         data=fp,
#         file_name="myfile.zip",
#         mime="application/zip"
#     )
