import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import kagglehub
import os
import random

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️"
)

st.title("🛣️ Road Damage Detection")

# Load Kaggle dataset
@st.cache_resource
def load_dataset():

    path = kagglehub.dataset_download(
        "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
    )

    image_path = os.path.join(
        path,
        "data",
        "images"
    )

    return image_path


# Load model
@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "road_damage_cnn.keras"
    )


image_folder = load_dataset()

model = load_model()

labels = {

0:"pothole",

1:"crack",

2:"manhole"

}


st.write(
"Testing directly from Kaggle dataset"
)

if st.button(
    "Predict Random Road Image"
):

    img_name = random.choice(
        os.listdir(image_folder)
    )

    img_path = os.path.join(
        image_folder,
        img_name
    )

    image = Image.open(
        img_path
    ).convert(
        "RGB"
    )

    st.image(
        image,
        caption=img_name,
        use_container_width=True
    )

    img = image.resize(
        (224,224)
    )

    img = np.array(
        img
    )/255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    pred = model.predict(
        img,
        verbose=0
    )

    pred_class = np.argmax(
        pred
    )

    confidence = np.max(
        pred
    )*100

    st.success(

f"Prediction: {labels[pred_class]}"

    )

    st.info(

f"Confidence: {confidence:.2f}%"

    )