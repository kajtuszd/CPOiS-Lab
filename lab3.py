import cv2
import math
import numpy
import matplotlib.pyplot as plt
import copy

'''
fourier transform

'''

img = cv2.imread("gull.png", 0)
dft = cv2.dft(numpy.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = numpy.fft.fftshift(dft)

magnitude_spectrum = 20*numpy.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


'''
inverse fourier transform

'''

rows, cols = img.shape
crow, ccol = int(rows/2) , int(cols/2)

mask = numpy.zeros((rows, cols,2), numpy.uint8)
mask[crow-30: crow+30, ccol-30: ccol+30] = 1

fshift = dft_shift*mask
f_ishift = numpy.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Transformed Image'), plt.xticks([]), plt.yticks([])
plt.show()


'''
filtration - low pass - averaging
'''

img1 = cv2.imread("ball.png", 0)

kernel = numpy.ones((5,5),numpy.float32)/25
dst = cv2.filter2D(img1,-1,kernel)

plt.subplot(121), plt.imshow(img1), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])

plt.show()

res = numpy.hstack((img1, dst))
cv2.imshow('Low-pass averaging filtration', res)
cv2.waitKey(0)

img1[:] += dst[:]
cv2.imshow('Sum', dst)
cv2.waitKey(0)


'''
filtration - high pass - Laplacian

'''

img2 = cv2.imread("ball.png", 0)

edges = cv2.Laplacian(img2, ddepth=-1, ksize=1, scale=1, delta=0,
                            borderType=cv2.BORDER_DEFAULT)

plt.subplot(121), plt.imshow(img2), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges), plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.show()

res = numpy.hstack((img2, edges))
cv2.imshow('High-pass filtration - Laplacian', res)
cv2.waitKey(0)

img2[:] += edges[:]
cv2.imshow('Sum', img2)
cv2.waitKey(0)
