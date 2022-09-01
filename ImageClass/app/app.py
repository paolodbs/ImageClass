import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64


url="http://127.0.0.1:8000/predict"

def send_data_to_api(uploaded_image):
    response = requests.post(url , files = {'file':uploaded_image.getvalue()})
    return response.json() # <--my prediction of the category of the image
uploaded_image = st.file_uploader("My Lovely Image")
#set_background('../background.png')

if uploaded_image is not None:
    #to read image as bytes:
    response = send_data_to_api(uploaded_image)
    st.image(uploaded_image)
    st.markdown(
        """
<style>
canvas {
max-width: 100%!important;
height: auto!important;
</style>
""",
        unsafe_allow_html=True
    )
    st.balloons()
    message = f'It´s clearly a  {response.get("prediction")} silly!'
    st.markdown(f"""
    #### <span style="color:blue">{message}</span>
  """,  unsafe_allow_html=True)
    col1 , col2 = st.columns(2)
    probabilities = pd.DataFrame(response['probabilities'] , index = [0]).T
    probabilities.columns = ['confidence']
    with col1:
        st.dataframe(probabilities.style.background_gradient(cmap = 'summer'))
    with col2:
        with plt.style.context('seaborn'):
            fig , ax = plt.subplots(nrows = 1 , ncols = 1 , figsize = (10,6))
            probabilities.plot(kind='bar' , ax = ax)
            st.pyplot(fig)
    #to convert to a string based io:
    #stringio = Stringio(uploaded_imsge.getvalue().decode("utf-8"))
    #st.write(stringio)
