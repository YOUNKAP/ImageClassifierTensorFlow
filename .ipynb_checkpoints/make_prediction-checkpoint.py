# TODO: Create the predict function

from PIL import Image
import numpy as np 
import tensorflow as tf
import json
image_size = 224
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
# TODO: Create the process_image function

from Process_image import process_image

# TODO: Create the predict function
def predict(image_path, model, top_k):
    #Load the image
    image = Image.open(image_path)
    #convert image to numpy array
    image_array = np.asarray(image)
    #Process the image 
    image_processed = process_image(image_array)
    #add extra dimension to the image
    image_batch_size = np.expand_dims(image_processed, axis=0)
    #Make the prediction
    predict = model.predict(image_batch_size)
    #Finds values and indices of the k top probabilities
    top_k_values, top_k_indices = tf.math.top_k(predict, top_k)
    top_k_classes = [class_names[str(index +1)] for index in top_k_indices.numpy()[0]]
    return top_k_values.numpy()[0], top_k_classes