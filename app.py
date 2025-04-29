import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Attempt to import Iterable from collections.abc; fallback to collections if ImportError occurs
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

# Hide Streamlit menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mango_model.h5')
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("# Mango Disease Detection")

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    predictions = import_and_predict(image, model)

    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
                   'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

    detected_disease = class_names[np.argmax(predictions)]

    # Show disease name below the image
    st.markdown(f"## **Detected Disease:** {detected_disease}")

    # Celebrate if it's healthy
    if detected_disease == 'Healthy':
        st.text(" The mango leaf is healthy! ")
