import numpy as np
from scipy import signal, misc, ndimage
import cv2
from matplotlib import pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
import jpeglab as jl
from DemoA import mse, psnr

orig = cv2.imread('image1.png')
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

y, cb, cr = jl.rgb2ycbcr(orig)

plt.figure(0)
plt.imshow(y, 'gray', clim=(0, 255))
plt.title('y')

plt.show(block=False)



plt.figure(1)
plt.imshow(cb, 'gray')
plt.title('cb')
cb2 = ndimage.zoom(cb, 0.5, order=3)
plt.figure(2)
plt.imshow(cb2, 'gray')
plt.title('cb2')

cbnew = ndimage.zoom(cb2, 2., order=3)
plt.figure(3)
plt.imshow(cbnew, 'gray')
plt.title('cbnew')

cr2 = ndimage.zoom(cr, 0.5, order=3)
plt.figure(4)
plt.imshow(cr2, 'gray')
plt.title('cr2')

crnew = ndimage.zoom(cr2, 2., order=3)
plt.figure(5)
plt.imshow(crnew, 'gray')

y2 = ndimage.zoom(y, 0.5, order=3)
plt.figure(6)
plt.imshow(y2, 'gray')
plt.title('y2')

ynew = ndimage.zoom(y2, 2., order=3)
plt.figure(7)
plt.imshow(ynew, 'gray')
plt.title('ynew')



print(f'PSNR y: {psnr(y, ynew)}')
print(f'PSNR cb: {psnr(cb, cbnew)}')
print(f'PSNR cr: {psnr(cr, crnew)}')

plt.show(block=True)