import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import imageio



#fig, axes = plt.subplots(nrows=2, ncols=2)
#fig.tight_layout()

graycmap = plt.get_cmap('gray', 256)
gray_vals = graycmap(np.arange(256))
gray_vals[200:] = [0, 0, 1, 1] #Blue
gray_vals[:50] = [0, 1, 0, 1] #Green
plt.register_cmap('ngray', graycmap.from_list('ngray', gray_vals))

Im = np.double(imageio.imread('Lab3bilder/baboon.tif'))
plt.subplot(1, 2, 1)
plt.imshow(Im, 'ngray', clim=(0,255))
plt.title('blue green img')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(Im, 'jet', clim=(0,255))
plt.title('jet')
plt.colorbar()


plt.show()

# Q5 color nose
#Answer:
# c) cyan