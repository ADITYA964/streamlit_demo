import keras
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

hash_funcs={'_thread.RLock' : lambda _: None, 
                '_thread.lock' : lambda _: None, 
                'builtins.PyCapsule': lambda _: None, 
                '_io.TextIOWrapper' : lambda _: None, 
                'builtins.weakref': lambda _: None,
                'builtins.dict' : lambda _:None}

def get_mean_std_per_batch(X):
    mean = np.mean(X)
    std = np.std(X)
    return mean, std    
        
def preprocess(image, H = 320, W = 320):
    """Load and preprocess image."""
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
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_coursera_model():
    """
    Model reference: Coursera AI for medicine
    """
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

    # create the base pre-trained model
    latest_iteration = st.empty()
    bar = st.progress(0)
    
    base_model = DenseNet121(include_top=False) #(weights='densenet.hdf5', include_top=False)
    latest_iteration.text("Loaded DenseNet121")
    bar.progress(1/19)
        
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    latest_iteration.text("Added a global spatial average pooling layer")
    bar.progress(2/19)
    
    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    latest_iteration.text("Added a logistic layer")
    bar.progress(3/19)
    
    model.load_weights("pretrained_model.h5")
    latest_iteration.text("Loaded pretrained weights")
    bar.progress(4/19)
    
    model._make_predict_function()
    latest_iteration.text("Compiled the model for prediction")
    bar.progress(5/19)
    
    funcs = []
    for i in range(len(labels)):
        output_with_batch_dim = model.output
        output_all_categories = output_with_batch_dim[0]
        y_c = output_all_categories[i]
        spatial_map_layer = model.get_layer('conv5_block16_concat').output
        grads_l = K.gradients(y_c, spatial_map_layer)
        grads = grads_l[0]
        spatial_map_and_gradient_function = K.function([model.input], [spatial_map_layer, grads])
        funcs.append(spatial_map_and_gradient_function)
        latest_iteration.text(f"Compiled the function for making heatmap of {labels[i]}")
        bar.progress((i+6)/19)
    
    session = K.get_session()
    print('Loaded')
    return model, funcs, session

@st.cache
def get_prediction(image, model):
    # return predicted labels 
    x = preprocess(image)
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    probas = model.predict(x)[0]
    return labels, probas

@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs=hash_funcs)
def get_heatmaps(image, model, funcs):

    def grad_cam(input_model, image, category_index, func):
        spatial_map_all_dims, grads_val_all_dims = func([image])
        spatial_map_val = spatial_map_all_dims[0]
        grads_val = grads_val_all_dims[0]
        weights = grads_val.mean(axis=0).mean(axis=0)
        cam = np.dot(spatial_map_val,weights)
        H, W = image.shape[1], image.shape[2]
        cam = np.maximum(cam, 0) # ReLU so we only get positive importance
        cam = Image.fromarray(cam)
        cam = np.asarray(cam.resize((W,H),Image.ANTIALIAS))
        #cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
        cam = cam / cam.max()
        return cam
    
    labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
                'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
    
    x = preprocess(image)
    heatmaps = []
    
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(14):
        # Update the progress bar with each iteration.
        latest_iteration.text(labels[i])
        bar.progress((i + 1)/14)
        heatmaps.append(grad_cam(model, x, i, funcs[i]))
    return heatmaps

