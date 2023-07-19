import re
import numpy as np
import os
import cv2
from flask import Flask, app, request, render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import preprocess_input

model = load_model(r"Updated-Xception-diabetic-retinopathy.h5")
model1 = load_model(r"VGG16.h5")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/vgg.html')
def vgg():
    return render_template('vgg.html')

@app.route('/index.html')
def home():
    return render_template("index.html")

@app.route('/result',methods = ["GET","POST"])
def res():
    if request.method == "POST":
        f=request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img = image.load_img(filepath,target_size=(299,299))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)

        index = ['No Diabetic Retinopathy', 'Mild Diabetic Retinopathy', 'Moderate Diabetic Retinopathy', 'Severe Diabetic Retinopathy', 'Proliferative Diabetic Retinopathy']
        result = str(index[ prediction[0]])
        print(result)
        return render_template('prediction.html',prediction=result)
 
@app.route('/vgg_result',methods = ["GET","POST"])
def alg():
   if request.method == "POST":
       f=request.files['image']
       basepath = os.path.dirname(__file__)
       filepath = os.path.join(basepath,'uploads',f.filename)
       f.save(filepath)
       
       img = image.load_img(filepath,target_size=(299,299))
       x = image.img_to_array(img)
       x = np.expand_dims(x,axis=0)
       img_data = preprocess_input(x)
       vgg_prediction = np.argmax(model1.predict(img_data), axis=1)

       index = ['No Diabetic Retinopathy', 'Mild Diabetic Retinopathy', 'Moderate Diabetic Retinopathy', 'Severe Diabetic Retinopathy', 'Proliferative Diabetic Retinopathy']
       vgg_result = str(index[ vgg_prediction[0]])
       print(vgg_result)
       return render_template('vgg.html',vgg_prediction=vgg_result)
   
if __name__ == "__main__":
    app.run()