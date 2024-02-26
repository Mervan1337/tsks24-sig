import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import imageio

Im = np.double(imageio.imread('Lab3bilder/baboon.tif'))

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.tight_layout()

plt.subplot(2, 2, 1)
plt.imshow(Im, 'gray', clim=(0,255))
plt.title('original image')
plt.colorbar()

def on_press(event):
    x = int(round(event.xdata));
    y = int(round(event.ydata));
    print(f"{x},{y} = {Im[y, x]}")

fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', on_press)

plt.subplot(2, 2, 2)
plt.imshow(Im, 'gray', clim=(50, 200))

plt.title('contrast image')
plt.colorbar()
#plt.show()
# Q1
print(Im)
print("min =", np.min(Im)) 
print("max =", np.max(Im)) 

# Answer:
# a)
# min =  2.0
# max =  207.0


# Q2
graycmap = plt.get_cmap('gray', 256)
gray_vals = graycmap(np.arange(256))
print(gray_vals)
print("\n\n")
#63,16 = 90.0

# Answer:
# b)  (X,Y) ≈ (62,16), index ≈ 84.

# Q3
gray_vals[200:] = [1, 0, 0, 1]
plt.register_cmap('ngray', graycmap.from_list('ngray', gray_vals))
print(gray_vals)

plt.subplot(2, 2, 3)
plt.imshow(Im, 'ngray', clim=(0, 255))
plt.title('red upper range image')
plt.colorbar()
plt.show()

# Answer:
# c) gray_vals has values between 0 and 1 instead of between 0 and 255.

# Q4
# Answer:
# c) Values ≥ 200 are shown in red.