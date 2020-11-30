import cv2
import numpy as np
import math
import copy
import matplotlib.pyplot as plt


def add_noise(image, mult):
    row = image.shape[0]
    col = image.shape[1]
    mean = 0
    var = 0.3
    sigma = var**0.05
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = copy.deepcopy(image)
    for i in range(row):
        for j in range(col):
            noisy[i][j] = image[i][j] + gauss[i][j]*mult
    return noisy


# '''
# low pass Gauss filter
# '''

# radius = 7

# img = cv2.imread("ball.png")

# img2 = add_noise(img,2)
# img5 = add_noise(img,5)
# img10 = add_noise(img,10)
# img20 = add_noise(img,20)


# img_gaussian2 = cv2.GaussianBlur(img2, (radius,radius), 0)
# cv2.imshow("Img with noise 2", np.hstack((img2, img_gaussian2)))

# img_gaussian5 = cv2.GaussianBlur(img5, (radius,radius), 0)
# cv2.imshow("Img with noise 5", np.hstack((img5, img_gaussian5)))

# img_gaussian10 = cv2.GaussianBlur(img10, (radius,radius), 0)
# cv2.imshow("Img with noise 10", np.hstack((img10, img_gaussian10)))

# img_gaussian20 = cv2.GaussianBlur(img20, (radius,radius), 0)
# cv2.imshow("Img with noise 20", np.hstack((img20, img_gaussian20)))

# cv2.waitKey()
# cv2.destroyAllWindows

# '''
# low pass median filter
# '''
# # median

# img = cv2.imread("marine.png")
# cv2.imshow("Original", img)
# noisy = add_noise(img, 10)
# cv2.imshow("Original with noise added", noisy)


# img_median = cv2.medianBlur(img, 5)
# cv2.imshow("Img median", img_median)

# img_median_noise = cv2.medianBlur(noisy, 5)
# cv2.imshow("Img median with noise", img_median_noise)

# cv2.waitKey(0)
# cv2.destroyAllWindows


# # gaussian 

# img = cv2.imread("marine.png")
# cv2.imshow("Original", img)
# noisy = add_noise(img, 10)
# cv2.imshow("Original with noise added", noisy)


# img_gaussian = cv2.GaussianBlur(img, (5,5), 0)
# cv2.imshow("Img Gaussian", img_gaussian)

# img_gaussian_noisy = cv2.GaussianBlur(noisy, (5,5), 0)
# cv2.imshow("Img Gaussian with noise", img_gaussian_noisy)

# cv2.waitKey(0)
# cv2.destroyAllWindows


'''
high pass filters
'''
img = cv2.imread("ball.png",0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

cv2.imshow("Original", img)
cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.imshow("Laplacian", laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows


'''
magnitudes of linear low and high pass filters
'''

plt.subplot(2,2,1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

radius = 5

img_gaussian = cv2.GaussianBlur(img, (radius,radius), 0)
img_median = cv2.medianBlur(img, 5)

plt.subplot(1,2,1), plt.imshow(img_gaussian, cmap='gray')
plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(img_median, cmap='gray')
plt.title('Median'), plt.xticks([]), plt.yticks([])
plt.show()
