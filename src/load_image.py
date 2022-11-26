"""Shows how to reaa an image from the disk.

- Uses matplotlib dn openCV to load the image.
- Both libs return the image as numpy array.
- Note that each lib loads the colors differently.
"""


import matplotlib.pyplot as plt

FILEPATH = './data/horse.jpg'

img_1 = plt.imread(FILEPATH)
print(type(img_1))
print(img_1.shape)
plt.imshow(img_1)
plt.show()

import cv2
img_2 = cv2.imread(FILEPATH, -1)
print(type(img_2))
print(img_2.shape)
plt.imshow(img_2)
plt.show()

