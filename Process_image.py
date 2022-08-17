# TODO: Create the process_image function
import tensorflow as tf
image_size = 224
def process_image(image):
    #Resize image
    image = tf.image.resize(image, (image_size, image_size))
    image/= 255.0
    image.numpy()
    return image