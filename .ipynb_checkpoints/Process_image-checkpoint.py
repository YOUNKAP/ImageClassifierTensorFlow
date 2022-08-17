# TODO: Create the process_image function
import tensorflow as tf
#Set image size
image_size = 224
def process_image(image):
    #Resize image
    image = tf.image.resize(image, (image_size, image_size))
    #Normalize the image
    image/= 255.0
    #Convert the tensor in numpy
    image.numpy()
    return image