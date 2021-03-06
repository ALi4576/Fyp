#Dependencies
import os
from flask import Flask, request, jsonify, flash, redirect, url_for,send_file
from werkzeug.utils import secure_filename
import traceback
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from keras.preprocessing import image
import tensorflow as tf
from flask_cors import CORS
import sys


UPLOAD_FOLDER = 'Uploaded'
# Input_Folder = 'woalsdnd-v-gan-eac27f2e9a3d/data/DRIVE/test/images'
# Output_folder = "woalsdnd-v-gan-eac27f2e9a3d/inference_outputs/DRIVE"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Your API definition
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['Input_Folder'] = Input_Folder
# app.config['Output_FOLDER'] = Output_folder
CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def prepare(filepath):
#     img_rows, img_cols = 200, 200
#     im = Image.open(filepath)   
#     img = im.resize((img_rows,img_cols))
#     gray = img.convert('L')
#     # IMG_SIZE=200
#     print(type(filepath))
#     # print(type(filepath))
#     # img_array=cv2.imread(filepath)
#     print(type(im))
#     # img_array/=255
#     # new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#     return np.array(gray).flatten()


@app.route('/predict', methods=['POST'])
def predict():
        try:
            if request.method == 'POST':
            # check if the post request has the file part
                if 'file' not in request.files:
                    flash('No file part')
                    return redirect(request.url)
            file = request.files['file']
            print(file)
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            images = image.load_img('Uploaded/' + filename, target_size=(200, 200))    
            x = image.img_to_array(images)
            x = tf.image.rgb_to_grayscale(x)
            x = np.expand_dims(x, axis=0)
            x = x/255.0
            predictions =loaded_model.predict(x)
            classes = np.argmax(predictions, axis = 1)
            return jsonify({'prediction': str(classes)})

        except:

            return jsonify({'trace': traceback.format_exc()})


# @app.route('/seg', methods=['POST'])
# def segment():
#         try:
#             if request.method == 'POST':
#             # check if the post request has the file part
#                 if 'file' not in request.files:
#                     flash('No file part')
#                     return redirect(request.url)
#             file = request.files['file']
#             # if user does not select file, browser also
#             # submit a empty part without filename
#             if file.filename == '':
#                 flash('No selected file')
#                 return redirect(request.url)
#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             filename = secure_filename(file.filename)
#             imag = Image.open('Uploaded/' + filename)
#             new_image = imag.resize((565, 584))
#             new_image.save("woalsdnd-v-gan-eac27f2e9a3d/data/DRIVE/test/images/" + filename)   
#             inference.inf()
#             os.remove("woalsdnd-v-gan-eac27f2e9a3d/data/DRIVE/test/images/" + filename)
#             full_filename = os.path.join(app.config['Output_FOLDER'], filename)
#             json_file = open('G:/Documents/University/python api/bin_class_model.json','r')
#             loaded_model_json = json_file.read()
#             json_file.close()
#             loaded_model = model_from_json(loaded_model_json)
#             loaded_model.load_weights("G:/Documents/University/python api/bin_class_weight.h5")
#             print ('Model loaded')
#             loaded_model.save('bin_class_weight.h5')
#             loaded_model = load_model('bin_class_weight.h5')
#             img = image.load_img('G:/Documents/University/python api/woalsdnd-v-gan-eac27f2e9a3d/inference_outputs/DRIVE/' + filename, target_size=(200,200))
#             X= image.img_to_array(img)
#             X= np.expand_dims(X, axis=0)
#             images = np.vstack([X])
#             val = loaded_model.predict(images)
#             if val == 0:
#                 return jsonify({'Prediction': 'no DR'})
#             else:
#                 print("pdr")
#                 return jsonify({'Prediction': 'PDR'})

#         except:
#             return jsonify({'trace': traceback.format_exc()})
 

# @app.route('/seg_res', methods=['POST'])
# def seg_res():
    # try:
    #     if request.method == 'POST':
    #     # check if the post request has the file part
    #         if 'file' not in request.files:
    #             flash('No file part')
    #             return redirect(request.url)
    #     file = request.files['file']
    #     # if user does not select file, browser also
    #     # submit a empty part without filename
    #     if file.filename == '':
    #         flash('No selected file')
    #         return redirect(request.url)
    #     if file and allowed_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['Output_FOLDER'], filename))
    #     filename = secure_filename(file.filename)
    #     json_file = open('G:/Documents/University/python api/bin_class_model.json','r')
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     loaded_model = model_from_json(loaded_model_json)
    #     loaded_model.load_weights("G:/Documents/University/python api/bin_class_weight.h5")
    #     print ('Model loaded')
    #     loaded_model.save('bin_class_weight.h5')
    #     loaded_model = load_model('bin_class_weight.h5')
    #     img = image.load_img('G:/Documents/University/python api/woalsdnd-v-gan-eac27f2e9a3d/inference_outputs/DRIVE/' + filename, target_size=(200,200))
    #     X= image.img_to_array(img)
    #     X= np.expand_dims(X, axis=0)
    #     images = np.vstack([X])
    #     val = loaded_model.predict(images)
    #     if val == 0:
    #         return jsonify({'Prediction': 'no DR'})
    #     else:
    #         print("pdr")
    #         return jsonify({'Prediction': 'PDR'})

    # except:
    #     return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    json_file = open('E:/Documents/University/python api/fypmodelali_final.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("E:/Documents/University/python api/fypmodelali_final.h5")
    print ('Model loaded')
    loaded_model.save('fypmodelali_final.h5')
    loaded_model = load_model('fypmodelali_final.h5')    
    app.run(port=7000,debug=True)