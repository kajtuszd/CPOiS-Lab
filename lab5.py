import cv2
import numpy as np
import matplotlib.pyplot as plt


'''
Thresholding
'''

def add_gauss_noise(img):
    gauss = np.random.normal(0, 1, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    img_gauss = cv2.add(img, gauss)
    return img_gauss

img = cv2.imread("ball.png")
img_gauss = add_gauss_noise(img)
cv2.imshow('ball.png with noise', img_gauss)
cv2.waitKey(0)

plt.hist(img.ravel(), 256, [0, 256])
plt.show()

img_low_pass = cv2.medianBlur(img, 5)
img_low_pass_noise = cv2.medianBlur(img_gauss, 5)

titles = ["Original image", "With noise", "filtered", "filtered noise"]

ret, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gauss, 50, 255, cv2.THRESH_BINARY)
ret, thresh3 = cv2.threshold(img_low_pass, 50, 255, cv2.THRESH_BINARY)
ret, thresh4 = cv2.threshold(img_low_pass_noise, 50, 255, cv2.THRESH_BINARY)

images = [thresh1, thresh2, thresh3, thresh4]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


'''
Edge operators
'''
# Canny

img = cv2.imread("ball.png", 0)
edges = cv2.Canny(img, 50, 150) # first & second threshold of hysteresis procedure
plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#Sobel
img = cv2.imread("ball.png", 0)
sobel_x = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)
sobel_y = cv2.Sobel(img, cv2.CV_8UC1, 0, 1)
sobel = cv2.add(sobel_x, sobel_y)

cv2.imshow("Sobel", sobel)
cv2.waitKey(0)

