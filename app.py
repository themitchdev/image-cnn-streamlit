import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import keras

from util import classify


# import model's training history
with open('History', "rb") as file_pi:
    history = pickle.load(file_pi)


st.set_page_config(page_title="Butterfly Inc", layout="wide")
st.title("🦋 Butterfly Inc Image Classifier 🦋")
st.subheader("A new Butterfly web application service to classify your butterfly specimen images with machine learning algorithm based on Convolutional Neural Network technique")
st.divider()


st.subheader("UPLOAD IMAGE OF YOUR BUTTERFLY SPECIMEN")
uploaded_file = st.file_uploader(
    "Choose a file", type=['jpeg', 'jpg', 'png]'], accept_multiple_files=False)


if 'img_dict' not in st.session_state:
    st.session_state.img_dict = []

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image)
    if st.button('Run Prediction'):
        label = classify(uploaded_file)[0]
        st.session_state.img_dict.append([uploaded_file, label])
        st.subheader("Your specimen: {}".format(label))


st.divider()

st.subheader("More info about this CNN's dataset below ⬇️")
with st.expander("Dataset for this model"):
    num_images = pd.read_csv("./file_out.csv")
    my_chart = num_images.label.value_counts()
    st.bar_chart(my_chart, width=720)


with st.expander("Prediction accuracy during training"):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig = plt.figure(1)
    st.pyplot(fig, use_container_width=False, dpi=100)

with st.expander("Confusion Matrix using a subset of validation images"):
    image = Image.open('./confusion_matrix.png')
    st.image(image, caption='75 Validation Images Use for the Consfusion Matrix')


with st.sidebar:
    st.title("Your Classification History")
    st.text("")
    for item in st.session_state.img_dict:
        st.sidebar.image(item[0])
        st.subheader(item[1])
        st.divider()
