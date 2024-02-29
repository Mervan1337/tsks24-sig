import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import imageio

Im = np.load('Lab3Bilder/pirat.npy')

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()

laplace = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])

"""
plt.subplot(2, 2, 1)
plt.imshow(Im, 'gray', clim=(0,255))
plt.title('original image')
plt.colorbar()"""

Imlaplace = signal.convolve2d(Im, laplace, 'same')
"""
plt.subplot(2, 2, 2)
plt.imshow(Imlaplace, 'gray', clim=(-100, 100))
plt.title('laplace image')
plt.colorbar()"""

ImHP = -Imlaplace
Imsharp = Im + ImHP
ImSharp2 = Im + 2*ImHP

plt.subplot(1, 2, 1)
plt.imshow(Imsharp, 'gray', clim=(0,255))
plt.title('sharp image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(ImSharp2, 'gray', clim=(0,255))
plt.title('sharper image')
plt.colorbar()
plt.show()

"""
plt.imshow(Im - ImHP, 'gray', clim=(0,255))
plt.title('wrong sign image')
plt.colorbar()
plt.show()"""


# QUESTION: 15
#Answer:
#15c) More blurry