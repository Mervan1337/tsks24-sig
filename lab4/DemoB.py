import numpy as np
from scipy import signal, misc, ndimage
from matplotlib import pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
import imageio
import cv2
import jpeglab as jl



orig = cv2.imread('image1.png')
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)


y, cb, cr = jl.rgb2ycbcr(orig)

plt.figure(2)
plt.imshow(y, 'gray', clim=(0, 255))
plt.title('y')
Y = cv2.dct(y)
plt.figure(3)
plt.imshow(np.log(np.abs(Y)+1),'gray')
plt.title('2d cosine transform of y')
plt.show(block=False)


Yq = np.zeros((512,768))
Yq[0:128,0:196] = np.round(Y[0:128,0:196])
plt.figure(4)
plt.imshow(np.log(np.abs(Yq)+1),'gray')
plt.title('2d cosine transform of y quantized')
yq = cv2.idct(Yq)
plt.figure(5)
plt.imshow(yq,'gray',clim=(0,255))
plt.title('coded image yq')

print(f'min: {np.min(Yq)} max: {np.max(Yq)}')


plt.show(block=True)


# QUESTION: 5
# a)


# QUESTION: 6
# a) eyes