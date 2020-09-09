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
    K.clear_session()
    """
    Model reference: Coursera AI for medicine
    """
    labels = ['心肥大', '肺気腫', '胸水', '横隔膜ヘルニア', '浸潤', '腫瘤', '結節', '無気肺',
              '気胸', '胸膜肥厚', '肺炎', '間質性肺炎', '肺水腫', 'コンソリデーション']

    # create the base pre-trained model
    latest_iteration = st.empty()
    bar = st.progress(0)
    
    base_model = DenseNet121(include_top=False) #(weights='densenet.hdf5', include_top=False)
    latest_iteration.text("Loaded DenseNet121")
    bar.progress(1/6)
        
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    latest_iteration.text("Added a global spatial average pooling layer")
    bar.progress(2/6)
    
    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    latest_iteration.text("Added a logistic layer")
    bar.progress(3/6)
    
    model.load_weights("pretrained_model.h5")
    latest_iteration.text("Loaded pretrained weights")
    bar.progress(4/6)
    
    model._make_predict_function()
    latest_iteration.text("Compiled the model for prediction")
    bar.progress(5/6)
    
    #funcs = []
    output_with_batch_dim = model.output
    output_all_categories = output_with_batch_dim[0]
    spatial_map_layer = model.get_layer('conv5_block16_concat').output
    grads = []
    for i in range(14):
        y_c = output_all_categories[i]
        grads_l = K.gradients(y_c, spatial_map_layer)[0]
        grads.append(grads_l)
    spatial_map_and_gradient_function = K.function([model.input], [spatial_map_layer] + grads)
    latest_iteration.text("Compiled heatmap function")
    bar.progress(100)
    session = K.get_session()
    print('Loaded')
    return model, spatial_map_and_gradient_function, session

@st.cache
def get_prediction(image, model):
    # return predicted labels 
    x = preprocess(image)
    labels = ['心肥大', '肺気腫', '胸水', '横隔膜ヘルニア', '浸潤', '腫瘤', '結節', '無気肺',
              '気胸', '胸膜肥厚', '肺炎', '間質性肺炎', '肺水腫', 'コンソリデーション']
    probas = model.predict(x)[0]
    return labels, probas

@st.cache(suppress_st_warning=True, allow_output_mutation=True, hash_funcs=hash_funcs)
def get_heatmaps(image, model, funcs):
    latest_iteration = st.empty()
    bar = st.progress(0)
    latest_iteration.text("Getting spatial map & gradients...")
    x = preprocess(image)
    results = funcs([x])
    spatial_map_all_dims = results[0]
    grads_val_all_dims = results[1:] # len = 14
    spatial_map_val = spatial_map_all_dims[0]
    bar.progress(1/15)
    
    heatmaps = []    
    labels = ['心肥大', '肺気腫', '胸水', '横隔膜ヘルニア', '浸潤', '腫瘤', '結節', '無気肺',
              '気胸', '胸膜肥厚', '肺炎', '間質性肺炎', '肺水腫', 'コンソリデーション']
    for i in range(14):
        grads_val = grads_val_all_dims[i][0]
        weights = grads_val.mean(axis=0).mean(axis=0)
        cam = np.dot(spatial_map_val,weights)
        H, W = x.shape[1], x.shape[2]
        cam = np.maximum(cam, 0) # ReLU so we only get positive importance
        cam = Image.fromarray(cam)
        cam = np.asarray(cam.resize((W,H),Image.ANTIALIAS))
        cam = cam / cam.max()
        heatmaps.append(cam)
        latest_iteration.text(labels[i])
        bar.progress((i + 2)/15)
    
    return heatmaps

