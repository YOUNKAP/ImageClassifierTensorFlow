# TODO: Create the process_image function
def process_image(image):
    #Resize image
    image = tf.image.resize(image, (image_size, image_size))
    image/= 255.0
    image.numpy()
    return image