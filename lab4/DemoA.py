import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
import imageio
import cv2
import jpeglab as jl


#Q1
# c) color R G B red hi lo lo green lo hi lo blue lo lo hi yellow hi hi lo color R G B cyan lo hi hi magenta hi lo hi white hi hi hi black lo lo lo

try:
    im1_bgr = cv2.imread('image1.png')

    if im1_bgr is None:
        raise Exception("Failed to load image. Check file path and format.")

    im1 = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2RGB)
    #plt.figure(0), plt.imshow(im1)

    im1r = im1[:, :, 0]
    im1g = im1[:, :, 1]
    im1b = im1[:, :, 2]
    """
    plt.figure(1), plt.imshow(im1r, 'gray'), plt.title('Red'), plt.colorbar()
    plt.figure(2), plt.imshow(im1g, 'gray'), plt.title('Green'), plt.colorbar()
    plt.figure(3), plt.imshow(im1b, 'gray'), plt.title('Blue'), plt.colorbar()"""

    y, cb, cr = jl.rgb2ycbcr(im1)
    """
    plt.figure(4), plt.imshow(y, 'gray', clim=(0, 255)), plt.title('y'), plt.colorbar()
    plt.figure(5), plt.imshow(cb, 'gray'), plt.title('cb'), plt.colorbar()
    plt.figure(6), plt.imshow(cr, 'gray'), plt.title('cr'), plt.colorbar()"""
    #plt.show()

except Exception as e:
    print(f"Error: {e}")
#Q2
#a)Most informationiscontainedintheY-component.

y, cb, cr = jl.rgb2ycbcr(im1)

plt.figure(2), plt.imshow(y, 'gray', clim=(0, 255))
plt.title('original (y)'), plt.colorbar()

X = 2**3
y2 = X*np.floor_divide(y,X)
plt.figure(3), plt.imshow(y2, 'gray', clim=(0, 255)), plt.title('y2 (5 bpp)'), plt.colorbar()

#Q3
# c) X = 2^3 = 8 to get 5 bpp

X = 2**2 # 6 bpp
y1 = X*np.floor_divide(y,X)
plt.figure(4), plt.imshow(y1, 'gray', clim=(0, 255)), plt.title('y1 (6 bpp)'), plt.colorbar()


X = 2**4 # 4 bpp
y3 = X*np.floor_divide(y,X)
plt.figure(5), plt.imshow(y3, 'gray', clim=(0, 255)), plt.title('y3 (4 bpp)'), plt.colorbar()
plt.show()

# QUESTION: 4
# c) 
# 6bpp⇒psnr=40.9dBand“good” 
# 5bpp⇒psnr=34.9dBand“half-good”