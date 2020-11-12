from cv2 import cv2
import numpy
import copy
import matplotlib.pyplot as plt

img1 = cv2.imread("ball.png", 0)
img2 = cv2.imread("mtf.png", 0)

cv2.imshow("image", img1)

#### 1)


'''
histogram equalization
'''

equ = cv2.equalizeHist(img1)
res = numpy.hstack((img1, equ))
cv2.imshow("Histogram equalization", res)
cv2.waitKey(0)

plt.figure(1)
plt.hist(img1.ravel(), 256, [0,256])
plt.figure(2)
plt.hist(res.ravel(), 256, [0,256])    
plt.show()

'''
contrast (histogram) stretching
'''

minmax_img = numpy.zeros((img1.shape[0], img1.shape[1]), dtype = 'uint8')

min_ = numpy.min(img1)
max_ = numpy.max(img1)
print(min_)
print(max_)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        minmax_img[i,j] = 255*((img1[i,j]-min_)/(max_-min_))

cv2.imshow('Histogram stretching', numpy.hstack((img1, minmax_img)))
cv2.waitKey(0)

plt.figure(1)
plt.hist(img1.ravel(), 256, [0,256])
plt.figure(2)
plt.hist(minmax_img.ravel(), 256, [0,256])    
plt.show()

#### 2)

img3 = copy.deepcopy(img2)
plt.hist(img3.ravel(), 256, [0,256])
plt.show()

minmax_img2 = numpy.zeros((img2.shape[0], img2.shape[1]), dtype = 'uint8')

min_ = numpy.min(img2)
max_ = numpy.max(img2)
print(min_)
print(max_)
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        minmax_img2[i,j] = 255*((img2[i,j]-min_)/(max_-min_))

equ = cv2.equalizeHist(minmax_img2)

equ[:] = equ[:]/8
equ[:] = equ[:]*8

img3[:] = img3[:]/8
img3[:] = img3[:]*8

both = numpy.hstack((img3, equ))
cv2.imshow("Quantization after histogram stretching and equalizing, Quantization", both)
cv2.waitKey(0)

plt.figure(1)
plt.hist(img3.ravel(), 256, [0,256])
plt.figure(2)
plt.hist(equ.ravel(), 256, [0,256])    
plt.show()