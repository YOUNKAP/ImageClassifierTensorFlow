import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import argparse
import json 


# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub 

from make_prediction import predict

from Process_image import process_image



parser = argparse.ArgumentParser()

parser.add_argument('--saved_model', type=str,  dest="save_model", default="best_model.h5" , help="Model to use for prediction")

parser.add_argument('--category_names', type=str , dest="category_names", default='label_map.json', help='Map cat file to name')

parser.add_argument('--top_k', type=int ,  dest="top_k", default=5,   help='Top 5 predicted classes')


parser.add_argument('--input_img', type=str ,  dest="input_img", default= "./test_images/wild_pansy.jpg", help='image to predict')


args ,_ = parser.parse_known_args() 

image_path = args.input_img

model_path = args.save_model

category_names = args.category_names

topk = args.top_k


model= tf.keras.models.load_model(
  model_path, 
  # `custom_objects` tells keras how to load a `hub.KerasLayer`
  custom_objects={'KerasLayer': hub.KerasLayer})

with open(category_names, 'r') as f:
    category_names= json.load(f)


predicted_probs, predicted_classes = predict(image_path, model, topk)


predicted_classes  =  predicted_classes[::-1]


predicted_probs = predicted_probs[::-1]


print("{} with a probability of {}".format(predicted_classes, predicted_probs))
 

#print("CONGRATULATIONS PROJECT N°2 COMPLETE, CARRY ON HARD WORK")
