import streamlit as st

from PIL import Image

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from geopy.geocoders import Nominatim

import folium
from folium import Marker

st.set_page_config(page_title = 'God\'s Eye', page_icon = ':sun_with_face:' , layout = 'wide')

hide_st_style = '''
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
'''

st.markdown(hide_st_style, unsafe_allow_html = True)

@st.experimental_memo(suppress_st_warning=True)
def function():

    def convert_address(address):
	    #Here we use Nominatin to convert address to a latitude/longitude coordinates"
	    geolocator = Nominatim(user_agent="my_app") #using open street map API 
	    Geo_Coordinate = geolocator.geocode(address)
	    lat = Geo_Coordinate.latitude
	    lon = Geo_Coordinate.longitude
	    #Convert the lat long into a list and store is as points
	    point = [lat, lon]
	    return point

    def display_map(point):
	    m = folium.Map(point, tiles='OpenStreetMap', zoom_start=10)

        # Add marker for Location
	    folium.Marker(location=point).add_to(m)
	
	    return st.markdown(m._repr_html_(), unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file")

    app_mode = st.selectbox("Choose area",
        ["Asia", "Africa", "Europe", "North America", "South America"])

    st.write('---')

    if app_mode == "Asia":

        if uploaded_file is not None:

            col1, col2 = st.columns([1, 2])
            with col1:        
                image = Image.open(uploaded_file)
                st.image(image)
            with col2:
                with st.spinner('Finding...'):
                    TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
                    LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
                    IMAGE_SHAPE = (321, 321)
        
                    classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                         input_shape=IMAGE_SHAPE+(3,),
                                                         output_key="predictions:logits")])
            
                    df = pd.read_csv(LABEL_MAP_URL)
                    label_map = dict(zip(df.id, df.name))
    
                    img = image.resize(IMAGE_SHAPE)
                    img = np.array(img)/255.0
    
                    img = img[np.newaxis, ...]
                    prediction = classifier.predict(img)

                    address = st.text_input("", label_map[np.argmax(prediction)])
                    coordinates = convert_address(address)
                    display_map(coordinates)

    elif app_mode == "Africa":

        if uploaded_file is not None:

            col1, col2 = st.columns([1, 2])
            with col1:        
                image = Image.open(uploaded_file)
                st.image(image)
            with col2:
                with st.spinner('Finding...'):
                    TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_africa_V1/1'
                    LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_africa_V1_label_map.csv'
                    IMAGE_SHAPE = (321, 321)
        
                    classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                         input_shape=IMAGE_SHAPE+(3,),
                                                         output_key="predictions:logits")])
            
                    df = pd.read_csv(LABEL_MAP_URL)
                    label_map = dict(zip(df.id, df.name))
    
                    img = image.resize(IMAGE_SHAPE)
                    img = np.array(img)/255.0
    
                    img = img[np.newaxis, ...]
                    prediction = classifier.predict(img)

                    address = st.text_input("", label_map[np.argmax(prediction)])
                    coordinates = convert_address(address)
                    display_map(coordinates)

    elif app_mode == "Europe":

        if uploaded_file is not None:

            col1, col2 = st.columns([1, 2])
            with col1:        
                image = Image.open(uploaded_file)
                st.image(image)
            with col2:
                with st.spinner('Finding...'):
                    TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_europe_V1/1'
                    LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_europe_V1_label_map.csv'
                    IMAGE_SHAPE = (321, 321)
        
                    classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                         input_shape=IMAGE_SHAPE+(3,),
                                                         output_key="predictions:logits")])
            
                    df = pd.read_csv(LABEL_MAP_URL)
                    label_map = dict(zip(df.id, df.name))
    
                    img = image.resize(IMAGE_SHAPE)
                    img = np.array(img)/255.0
    
                    img = img[np.newaxis, ...]
                    prediction = classifier.predict(img)

                    address = st.text_input("", label_map[np.argmax(prediction)])
                    coordinates = convert_address(address)
                    display_map(coordinates)

    elif app_mode == "North America":

        if uploaded_file is not None:

            col1, col2 = st.columns([1, 2])
            with col1:        
                image = Image.open(uploaded_file)
                st.image(image)
            with col2:
                with st.spinner('Finding...'):
                    TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_north_america_V1/1'
                    LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_north_america_V1_label_map.csv'
                    IMAGE_SHAPE = (321, 321)
        
                    classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                         input_shape=IMAGE_SHAPE+(3,),
                                                         output_key="predictions:logits")])
            
                    df = pd.read_csv(LABEL_MAP_URL)
                    label_map = dict(zip(df.id, df.name))
    
                    img = image.resize(IMAGE_SHAPE)
                    img = np.array(img)/255.0
    
                    img = img[np.newaxis, ...]
                    prediction = classifier.predict(img)

                    address = st.text_input("", label_map[np.argmax(prediction)])
                    coordinates = convert_address(address)
                    display_map(coordinates)

    elif app_mode == "South America":

        if uploaded_file is not None:

            col1, col2 = st.columns([1, 2])
            with col1:        
                image = Image.open(uploaded_file)
                st.image(image)
            with col2:
                with st.spinner('Finding...'):
                    TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_south_america_V1/1'
                    LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_south_america_V1_label_map.csv'
                    IMAGE_SHAPE = (321, 321)
        
                    classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                         input_shape=IMAGE_SHAPE+(3,),
                                                         output_key="predictions:logits")])
            
                    df = pd.read_csv(LABEL_MAP_URL)
                    label_map = dict(zip(df.id, df.name))
    
                    img = image.resize(IMAGE_SHAPE)
                    img = np.array(img)/255.0
    
                    img = img[np.newaxis, ...]
                    prediction = classifier.predict(img)

                    address = st.text_input("", label_map[np.argmax(prediction)])
                    coordinates = convert_address(address)
                    display_map(coordinates)

function()

st.experimental_memo.clear()
