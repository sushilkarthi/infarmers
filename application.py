from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


import cv2
# set the matplotlib backend so figures can be saved in the background

from keras import backend as K





# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
import tarfile

#tar_path = 'models/model.tar.gz'
MODEL_PATH = 'models/multi_class_model_2.h5'

#tar = tarfile.open(tar_path, "r:gz")
#tar.extractall()
#tar.close()


# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    image = cv2.resize(img, (64,64))
    image = np.array(image, dtype='float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    
    #x = np.array(img, dtype='float32') / 255.0
    #x = np.expand_dims(image, axis=0)

    # Preprocessing the image
 

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    lb = pickle.load(open('models/label_transform.pkl', 'rb'))
    preds = model.predict(image)
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    print(label)
    return label


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        print(result)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
