import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf


def conv_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x


def decoder_block(inputs, skip, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x


def build_effienet_unet(input_shape):
    """ Input """
    inputs = tf.keras.layers.Input(input_shape)

    """ Pre-trained Encoder """
    encoder = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = encoder.get_layer("input_1").output  ## 256
    s2 = encoder.get_layer("block2a_expand_activation").output  ## 128
    s3 = encoder.get_layer("block3a_expand_activation").output  ## 64
    s4 = encoder.get_layer("block4a_expand_activation").output  ## 32

    """ Bottleneck """
    b1 = encoder.get_layer("block6a_expand_activation").output  ## 16

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## 32
    d2 = decoder_block(d1, s3, 256)  ## 64
    d3 = decoder_block(d2, s2, 128)  ## 128
    d4 = decoder_block(d3, s1, 64)  ## 256

    """ Output """
    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d4)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_UNET")
    return model


input_shape = (256, 256, 3)
effienet_Unet_model = build_effienet_unet(input_shape)
effienet_Unet_model.load_weights("effienet_Unet_brain_final")


def load_preprocess_image(img):
    im = Image.open(img)
    image = np.array(im)
    image = image / 256.0
    return image


def predict_segmentation_mask(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    img -= img.mean()
    img /= img.std()
    X = np.empty((1, 256, 256, 3))
    X[0,] = img
    predict = effienet_Unet_model.predict(X)
    return predict.reshape(256, 256, 3)


def plot_MRI_predicted_mask(original_img, predicted_mask):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
    axes[0].imshow(original_img)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('Original MRI')
    axes[1].imshow(predicted_mask)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('Predicted Mask')
    fig.tight_layout()
    filename = 'pair' + str(random.randint(100, 1000)) + str(random.randint(100, 1000)) + '.png'
    plt.savefig(filename)
    return filename 


def final_fun_1(image_path):
    image = load_preprocess_image(image_path)
    mask = predict_segmentation_mask(image_path)
    combined_img = plot_MRI_predicted_mask(original_img=image, predicted_mask=mask)
    return combined_img


if uploaded_file is not None:
    try:
        image_path = uploaded_file
        combined_img = final_fun_1(image_path)
        st.image(combined_img)

    except Exception as e:
        st.error("An error occurred while processing the image: {}".format(e)) 
