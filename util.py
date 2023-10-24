from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pickle

#import model
model = tf.keras.models.load_model("./model2.tf")

# import training set class indices dict
with open('classes', "rb") as file_pi:
    classes = pickle.load(file_pi)


def classify(u_image):
    img = image.load_img(u_image)
    img = image.img_to_array(img)
    img = np.array([img])
    img = img.astype('float32')/255.
    index = model.predict(img).argmax(axis=-1)[0]
    return [k for k, v in classes.items() if v == index]
