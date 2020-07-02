import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from utils import load_coursera_model, get_prediction, get_heatmaps
import numpy as np
plt.rcParams['figure.figsize'] = [14, 10]

st.title("Image Classification with CheXNet")
st.header("Multi-diseases detection Example")
st.text("Upload Chest X-ray Image for multi-diseases detection")


uploaded_file = st.file_uploader("Choose data to input (only JPG, JPEG or PNG)")

if uploaded_file is not None:
    # Upload image and confirm
    image = Image.open(uploaded_file)
    shape = np.asarray(image).shape
    if len(shape) != 3:
        st.write("Your image is gray-scale image, you need to input color image!!")
    else:
        st.image(image, caption='Original Chest X-ray', use_column_width=True)
        # Model loading
        st.write("Loading pretrained model...")
        model = load_coursera_model(image) 
        st.write("Model loaded!!")

        if model is not None:
            st.write("")
            st.write("Classifying ...")
            labels, probas = get_prediction(image, model)
            probas = list(probas)
            prediction = {l:p for l,p in zip(labels,probas)}
            prediction = sorted(prediction.items(), key=lambda x:x[1], reverse=True)
            sorted_labels = [x[0] for x in prediction]
            sorted_probas = [x[1] for x in prediction]
            st.write("Done!!")
            if probas is not None:
                X = range(len(probas))
                plt.title("Multi-diseases detection")
                plt.bar(X, sorted_probas)
                plt.xticks(X, sorted_labels, rotation=45)
                st.pyplot()

                st.write("")
                st.write("Making heatmaps...")
                heatmaps = get_heatmaps(image, model)
                st.write("Done!!")
                # default heatmap is set to most likely disease
                label = st.selectbox(
                    'Choose disease to explore',
                    sorted_labels)

                if label is not None:
                    st.write("")
                    st.write(label + " is selected")
                    i = labels.index(label)
                    heatmap = heatmaps[i]
                    plt.title("Heatmap of "+ label)
                    plt.axis('off')
                    plt.imshow(ImageOps.fit(image, (320,320), Image.ANTIALIAS), cmap='gray')
                    plt.imshow(heatmap, cmap='magma', alpha=min(0.5, probas[i]))
                    st.pyplot()