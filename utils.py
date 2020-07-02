import keras
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
import streamlit as st

def get_mean_std_per_batch(X):
    mean = np.mean(X)
    std = np.std(X)
    return mean, std    
        
def preprocess(image, H = 320, W = 320):
    """resize and normalize image."""
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, H, W, 3), dtype=np.float32)
    #image sizing
    size = (H, W)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    x = np.asarray(image)[:,:,:3]

    # normalize
    mean, std = get_mean_std_per_batch(x)
    x = x - mean
    x = x / std
    data[0] = x
    return data

# LOAD COURSERA MODEL
@st.cache 
def load_coursera_model(image):
    """
    Model reference: Coursera AI for medicine
    """
    print(type(image))
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

    # create the base pre-trained model
    base_model = DenseNet121(include_top=False) #(weights='densenet.hdf5', include_top=False)
    print("Loaded DenseNet")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    print("Added layers")

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights("pretrained_model.h5")
    print("Loaded Weights")
    return model

@st.cache
def get_prediction(image, model):
    """return labels and predicted values of probability"""
    x = preprocess(image)
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    probas = model.predict(x)[0]
    return labels, probas

@st.cache(suppress_st_warning=True) 
def get_heatmaps(image, model):
    """return heatmaps of 14 common diseases"""
    x = preprocess(image)
    heatmaps = []
    
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(14):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Disease type: {i+1}')
        bar.progress((i + 1)/14)
        heatmaps.append(grad_cam(model, x, i))
    return heatmaps

def grad_cam(input_model, image, category_index, layer_name='conv5_block16_concat'):
    """
    Reference: Coursera AI for treatment
    GradCAM method for visualizing input saliency.
    Args:
    input_model (Keras.model): model to compute cam for
    image (tensor): input to model, shape (1, H, W, 3)
    category_index (int): class to compute cam with respect to
    layer_name (str): relevant layer in model
    Return:
    cam ()
    """
    output_with_batch_dim = input_model.output
    output_all_categories = output_with_batch_dim[0]
    y_c = output_all_categories[category_index]
    spatial_map_layer = input_model.get_layer(layer_name).output
    grads_l = K.gradients(y_c, spatial_map_layer)
    grads = grads_l[0]
    spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])
    spatial_map_val = spatial_map_all_dims[0]
    grads_val = grads_val_all_dims[0]
    weights = grads_val.mean(axis=0).mean(axis=0)
    cam = np.dot(spatial_map_val,weights)
    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0) # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()
    return cam
