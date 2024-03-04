import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import imageio


help(signal.convolve2d)
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()
Im = np.double(imageio.imread('Lab3bilder/baboon.tif'))
plt.subplot(2,2,1)
plt.imshow(Im,'gray',clim=(0,255))
plt.title('original')
plt.colorbar()


b = np.array([0.5,0.5])
b2 = np.convolve(b,b).reshape(1,3)
aver = signal.convolve2d(b2,b2.T)

Imaver = signal.convolve2d(Im, aver, 'same')
plt.subplot(2, 2, 2)
plt.imshow(Imaver, 'gray', clim=(0,255))
plt.title('convolved')
plt.colorbar()

Imaver_ = signal.convolve2d(Im, aver * 2, 'same')
plt.subplot(2, 2, 3)
plt.imshow(Imaver_, 'gray', clim=(0,255))
plt.title('8')
plt.colorbar()

#Imaver = signal.convolve2d(Imaver, aver, 'same')
Imaver = signal.convolve2d(Imaver, aver, 'same')
plt.subplot(2, 2, 4)
plt.imshow(Imaver, 'gray', clim=(0,255))
plt.title('3')
plt.colorbar()


plt.show()

# Q6
# Answer:
# b) They become blurred.

#help(signal.convolve2d)

# Q7
# Answer:
# c) Before convolution, the image is repeated in the x- and y-directions.

# Q8
# Answer:
# c) The resulting image gets more and more blurred.

# Q9
# Answer:
# b) The resulting image is still blurred, but also darker

