import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import imageio

Im = np.load('Lab3bilder/pattern.npy')

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

plt.subplot(2, 2, 1)
plt.imshow(Im, 'gray', clim=(-1,1))
plt.title('original image')

Imsample = Im[::2,::2]
plt.subplot(2, 2, 2)
plt.imshow(Imsample, 'gray', clim=(-1,1))
plt.title('downsampled image')

# QUESTION: 16
# a

b = np.array([0.5, 0.5])
b2 = np.convolve(b, b).reshape(1, 3)
aver = signal.convolve2d(b2, b2.T)

Imaver = Im
for _ in range(4):
    Imaver = signal.convolve2d(Imaver,aver,'same')

plt.subplot(2, 2, 3)
plt.imshow(Imaver, 'gray', clim=(-1,1))
plt.title('LP filtered image')

Imsample = Imaver[::2,::2]
plt.subplot(2, 2, 4)
plt.imshow(Imsample, 'gray', clim=(-1,1))
plt.title('LP-filtered & downsampled')
plt.show()


# QUESTION: 17
# 4