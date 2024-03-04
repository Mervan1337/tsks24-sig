import numpy as np
from scipy import signal, misc, ndimage
from matplotlib import pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
import imageio
import cv2
import jpeglab as jl



im1_bgr = cv2.imread('image1.png')

im1 = cv2.cvtColor(im1_bgr,cv2.COLOR_BGR2RGB)
y, cb, cr = jl.rgb2ycbcr(im1)
plt.figure(0), plt.imshow(y, 'gray', clim=(0, 255))
plt.title('original'), plt.colorbar()

Y = cv2.dct(y)
plt.figure(1), plt.imshow(np.log(np.abs(Y)+1),'gray')
plt.title('2d cosine transform'), plt.colorbar()

Yq = np.zeros((512,768))
Yq[0:128,0:196] = np.round(Y[0:128,0:196])
plt.figure(2), plt.imshow(np.log(np.abs(Yq)+1),'gray')
plt.title('2d cosine transform (take 1/16 coeffs)'), plt.colorbar()
yq = cv2.idct(Yq)
plt.figure(3), plt.imshow(yq,'gray', clim=(0,255))
plt.title('coded image (yq)'), plt.colorbar()
#plt.show()

print(f"Yq min = {np.min(Yq)} and Yq max = {np.max(Yq)}")

# QUESTION: 7
# b)18bits(218=131072) isneededtorepresentthelargestvalue(68923).


# QUESTION: 8
# c)18/16=1.125bpp,psnr=31.0dB,“worsethanhalf-good”


# QUESTION: 9
#b) Applying the fft2 and taking the lower right part gives the dct.


# Q 10 
# b)The dft othen gives asharp horizonal and/or vertical line.This is avoided with the dct. Consequently, the dct is easier to compress.

#print(f"Maximum difference: {np.max(np.abs(A-BPART))}")

# Q 11 
# a)The maximum is a very small positive value and the minimum is avery small negative value.

y, cb, cr = jl.rgb2ycbcr(im1)
plt.figure(0), plt.imshow(y, 'gray', clim=(0, 255))
plt.title('original'), plt.colorbar()

Yb = jl.bdct(y, (8, 8))
ulim = np.max(np.abs(Yb))/10
plt.figure(1), plt.imshow(np.abs(Yb), 'gray', clim=(0, ulim))
plt.title('bdct transformed'), plt.colorbar()

yn = jl.ibdct(Yb, (8, 8), (512, 768))
plt.figure(2), plt.imshow(yn, 'gray', clim=(0, 255))
plt.title('inverse transformed'), plt.colorbar()

#print(f"Maximum difference: {np.max(np.abs(y-yn))}")

# Q 12 
# c

Yb = jl.bdct(y, (8, 8))
Ybq = np.zeros(Yb.shape)
Ybq[(0, 1, 8, 9), :] = np.round(Yb[(0, 1, 8, 9), :])
yq2 = jl.ibdct(Ybq, (8, 8), (512, 768))
plt.figure(3), plt.imshow(yq2, 'gray', clim=(0, 255))
plt.title('coded image 4 coefficients'), plt.colorbar()


# Q13
# c

print(f"Ybq min = {np.min(Ybq)} and Ybq max = {np.max(Ybq)}")


# Q14: 
# a)The maximum is 1880 and the minimum −470.Thus 12bits i.e. asign bit plus 11bits(211=2048) are sufficient.

# Q 15
# a

# Q 16
# a

# Q 17
# a