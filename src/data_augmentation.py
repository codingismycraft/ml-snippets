"""Used when we do not have enough images.

- Helps to avoid model over-fitting.
- Makes data loading a little faster

In this snip we can see how we can load the images from the disk
to the memory and then create the image generator that will take
care of the resizing, the rescaling and the image distortion.
"""

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator1 = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
)

generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

TRAIN_IMGS = "./data"

import os
import cv2

path = "./data"
X = []
Y = []
for e in os.listdir(path):
    img = cv2.imread(os.path.join(path, e))
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(grayImage.shape)
    resized = cv2.resize(grayImage, (256, 256), interpolation=cv2.INTER_AREA)
    resized = np.resize(resized, resized.shape + (1,))
    X.append(resized)
    Y.append(1)

import matplotlib.pyplot as plt

train_generator = generator.flow(np.array(X), np.array(Y), batch_size=2)
img = next(train_generator)
plt.imshow(img[0][0])
plt.show()

i = 0
for img in train_generator:
    i += 1
    if i > 100:
        break
    plt.imshow(img[0][0])
    plt.show()

    print(img[0][0])
