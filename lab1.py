# Kajetan Zdanowicz 248933
# CPOIS LAB SR/TP 7:30

# Program realizujący kwantyzację, dyskretyzację i interpolację wybranego obrazu

import cv2
import numpy
import copy 
import math

img1 = cv2.imread("airport.png")

height = img1.shape[0]
width = img1.shape[1]


cv2.imshow("Original", img1)
cv2.waitKey(0)

# discretization

imgResize2 = cv2.resize(img1, (int(width/2), int(height/2)))
imgResize3 = cv2.resize(img1, (int(width/3), int(height/3)))
imgResize4 = cv2.resize(img1, (int(width/4), int(height/4)))
imgResize8 = cv2.resize(img1, (int(width/8), int(height/8)))
imgcpy = copy.deepcopy(imgResize8)
imgResize16 = cv2.resize(img1, (int(width/16), int(height/16)))

imgResize2 = cv2.resize(imgResize2, (width, height))
imgResize3 = cv2.resize(imgResize3, (width, height))
imgResize4 = cv2.resize(imgResize4, (width, height))
imgResize8 = cv2.resize(imgResize8, (width, height))
imgResize16 = cv2.resize(imgResize16, (width, height))

cv2.imshow("Image", img1)
cv2.imshow("Discretised image 2", imgResize2)
cv2.imshow("Discretised image 3", imgResize3)
cv2.imshow("Discretised image 4", imgResize4)
cv2.imshow("Discretised image 8", imgResize8)
cv2.imshow("Discretised image 16", imgResize16)
cv2.waitKey(0)

# interpolation
cubic = cv2.resize(imgcpy, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
nearest = cv2.resize(imgcpy, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
linear = cv2.resize(imgcpy, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)

cv2.imshow(" cubic", cubic)
cv2.imshow(" nearest", nearest)
cv2.imshow(" linear", linear)
cv2.imshow("Image", imgResize8)
cv2.waitKey(0)

# quantization

imgQuant2 = copy.deepcopy(img1)
imgQuant2[:] = imgQuant2[:]/2
imgQuant2[:] = imgQuant2[:]*2

imgQuant16 = copy.deepcopy(img1)
imgQuant16[:] = imgQuant16[:]/16
imgQuant16[:] = imgQuant16[:]*16

imgQuant64 = copy.deepcopy(img1)
imgQuant64[:] = imgQuant64[:]/64
imgQuant64[:] = imgQuant64[:]*64
imgQuant128 = copy.deepcopy(img1)
imgQuant128[:] = imgQuant128[:]/128
imgQuant128[:] = imgQuant128[:]*128


cv2.imshow("Image", img1)
cv2.imshow("Quantized image 2", imgQuant2)
cv2.imshow("Quantized image 16", imgQuant16)
cv2.imshow("Quantized image 64", imgQuant64)
cv2.imshow("Quantized image 128", imgQuant128)
cv2.waitKey(0)
