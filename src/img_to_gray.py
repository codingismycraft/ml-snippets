"""Converts an image to monochrome"""

import cv2
import matplotlib.pyplot as plt

FILEPATH = './data/horse.jpg'

originalImage = cv2.imread(FILEPATH)
print(originalImage.shape)

# This is how you can conver the image to gray.
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
print(grayImage.shape)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255,
                                             cv2.THRESH_BINARY)

print(blackAndWhiteImage.shape)

plt.imshow(originalImage)
plt.show()

plt.imshow(blackAndWhiteImage)
plt.show()

plt.imshow(grayImage, cmap='gray')
plt.show()


