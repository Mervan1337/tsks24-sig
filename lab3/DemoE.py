import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import imageio


fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
Im = np.double(plt.imread('Lab3bilder/baboon.tif'))
plt.subplot(2,2,1)
plt.imshow(Im,'gray',clim=(0,255))
plt.title('original image')
plt.colorbar()

b = np.array([0.5,0.5])
b2 = np.convolve(b,b).reshape(1,3)
d = np.array([1,-1.0])
cd = np.convolve(b,d).reshape(1,3)
sobelx = np.array([[1.0,0.0,-1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) /8
Imsobelx = signal.convolve2d(Im,sobelx,'same')

plt.subplot(2,2,3)
plt.imshow(Imsobelx,'gray',clim=(-128,127))
plt.title('sobelx image')
plt.colorbar()


sobely = np.array([[1.0,2.0,1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) /8
Imsobely = signal.convolve2d(Im,sobely,'same')

plt.subplot(2,2,4)
plt.imshow(Imsobely,'gray',clim=(-128,127))
plt.title('sobely image')
plt.colorbar()

im2 = np.sqrt(np.square(Imsobelx) + np.square(Imsobely))
plt.subplot(2,2,2)
plt.imshow(im2,'gray',clim=(0,255))
plt.title('magngrad image')
plt.colorbar()


plt.show()

