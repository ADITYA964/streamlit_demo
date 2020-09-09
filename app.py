import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from utils1 import load_coursera_model, get_prediction, get_heatmaps
import numpy as np
from keras import backend as K
plt.rcParams['figure.figsize'] = [14, 10]

def main():
    st.title("APP")
    st.header("CheXNet. ディープラーニングを用いた胸部X線撮影における放射線技師レベルの肺炎検出")
    st.text("Task: 胸部X線 > 複数疾患検出")
    st.text("Department: 呼吸器内科　循環器内科　感染症内科　放射線科")
    st.text("Citation: arXiv:1711.05225v3")
    st.write("Set up...")
    model, funcs, session = load_coursera_model() 
    st.write("Done!!")
        
    option = st.selectbox(
                            '',
                            ['Choose demo data or your data','Use Demo data','Use your data'])
    if option == 'Choose demo data or your data':
        st.write('')

    else:    
        if option == 'Use Demo data':
            image = Image.open('test.png')
            st.image(image, caption='Original Chest X-ray (label = Mass)', use_column_width=True)
                
            if model is not None:
                st.write("")
                st.write("Classifying ...")
                K.set_session(session)
                labels, probas = get_prediction(image, model)
                probas = list(probas)
                prediction = {l:p for l,p in zip(labels,probas)}
                prediction = sorted(prediction.items(), key=lambda x:x[1], reverse=True)
                sorted_labels = [x[0] for x in prediction]
                sorted_probas = [x[1] for x in prediction]
                st.write("Done!!")
                if probas is not None:
                    X = range(len(probas))
                    for label in sorted_labels:
                        i = labels.index(label)
                        st.write(label+f": {probas[i]:.3f}")
                    
                    st.write("")
                    st.write("Making heatmaps...")
                    heatmaps = get_heatmaps(image, model, funcs)
                    st.write("Done!!")
                    for label in sorted_labels:
                        i = labels.index(label)
                        st.write("Heatmap - "+ label + f": {probas[i]:.3f}")
                        heatmap = heatmaps[i]
                        #plt.title("Heatmap - "+ label + f": {probas[i]:.3f}",fontsize=20)
                        plt.axis('off')
                        plt.imshow(ImageOps.fit(image, (320,320), Image.ANTIALIAS), cmap='gray')
                        plt.imshow(heatmap, cmap='magma', alpha=min(0.5, probas[i]))
                        st.pyplot()
            
        else:
            uploaded_file = st.file_uploader("Choose data to input (only JPG, JPEG or PNG)")

            if uploaded_file is not None:
                    # Upload image and confirm
                image = Image.open(uploaded_file)
                shape = np.asarray(image).shape
                print(shape)
                if len(shape) != 3:
                    st.write("Your image is gray-scale image, you need to input color image!!")
                else:
                    st.image(image, caption='Original Chest X-ray', use_column_width=True)
                        # Model loading
                
                    if model is not None:
                        K.set_session(session)
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
                            for label in sorted_labels:
                                i = labels.index(label)
                                st.write(label+f": {probas[i]:.3f}")

                            st.write("")
                            st.write("Making heatmaps...")
                            heatmaps = get_heatmaps(image, model, funcs)
                            st.write("Done!!")
                            for label in sorted_labels:
                                i = labels.index(label)
                                st.write("Heatmap - "+ label + f": {probas[i]:.3f}")
                                heatmap = heatmaps[i]
                                #plt.title("Heatmap - "+ label + f": {probas[i]:.3f}",fontsize=20)
                                plt.axis('off')
                                plt.imshow(ImageOps.fit(image, (320,320), Image.ANTIALIAS), cmap='gray')
                                plt.imshow(heatmap, cmap='magma', alpha=min(0.5, probas[i]))
                                st.pyplot()  
                    
if __name__=="__main__":
    main()
