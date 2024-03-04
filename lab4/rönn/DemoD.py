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



Yb = jl.bdct(y, (8, 8))
ulim = np.max(np.abs(Yb))/10
#plt.figure(3)
#plt.imshow(np.abs(Yb),'gray', clim=(0, ulim))
#plt.title('bdct transform of y')

#plt.show(block=False)

yn = jl.ibdct(Yb, (8, 8), y.shape)
#plt.figure(4)
#plt.imshow(yn,'gray',clim=(0,255))
#plt.title('inverse transformaed image yn')

"""print('MSE:', mse(y,yn))
print('PSNR:', psnr(y,yn))
print('maximum diff: ', np.max(np.abs(y-yn)))"""

#plt.show(block=False)

Yb = jl.bdct(y, (8, 8))
Ybq = np.zeros_like(Yb)
Ybq[(0, 1, 8, 9), :] = np.round(Yb[(0, 1, 8, 9), :])
yq2 = jl.ibdct(Ybq, (8, 8), (512, 768))
#plt.figure(5)
#plt.imshow(yq2, 'gray', clim=(0, 255))
#plt.title('coded image yq2 4 coefficients')

"""
print(f'min: {np.min(Ybq)} max: {np.max(Ybq)}')

print('MSE:', mse(yn,yq2))
print('PSNR:', psnr(yn,yq2))"""

#plt.show(block=False)


Yb9 = jl.bdct(y, (8, 8))
Ybq9 = np.zeros_like(Yb)
Ybq9[(0, 1, 2, 8, 9, 10, 16, 17, 18), :] = np.round(Yb[(0, 1, 2, 8, 9, 10, 16, 17, 18), :])
yq29 = jl.ibdct(Ybq9, (8, 8), (512, 768))
#plt.figure(5)
#plt.imshow(yq2, 'gray', clim=(0, 255))
#plt.title('coded image yq2 9 coefficients')
"""
print("YBQ9-------------------------")
print(f'min: {np.min(Ybq9)} max: {np.max(Ybq9)}')
print('MSE:', mse(yn,yq29))
print('PSNR:', psnr(yn,yq29))"""

#plt.show(block=True)


Q1 = 50
Ybq = jl.bquant(Yb, Q1)
Ybr = jl.brec(Ybq, Q1)
yr = jl.ibdct(Ybr, (8, 8), (512, 768))
#plt.figure(1)
#plt.imshow(yr, 'gray', clim=(0, 255))
#plt.title('reconstructed image Q1=50')

"""print('Q1 = 50 -------------------------')
print('MSE:', mse(y, yr))
print('PSNR:', psnr(y, yr))"""

Q12 = 20
Ybq20 = jl.bquant(Yb, Q12)
Ybr20 = jl.brec(Ybq20, Q12)
yr20 = jl.ibdct(Ybr20, (8, 8), (512, 768))
#plt.figure(2)
#plt.imshow(yr20, 'gray', clim=(0, 255))
#plt.title('reconstructed image Q1=20')

"""print('Q1 = 20 -------------------------')
print('MSE:', mse(y, yr20))
print('PSNR:', psnr(y, yr20))"""


#plt.show(block=False)

Qm = jl.jpgqmtx()
Qm.reshape(8, 8)
#print(Qm)

JPEGMEAN = np.mean(Qm)
print(JPEGMEAN)

Q2 = 2.8
Ybq = jl.bquant(Yb, jl.jpgqmtx()*Q2)
Ybr = jl.brec(Ybq, jl.jpgqmtx()*Q2)
yr = jl.ibdct(Ybr, (8, 8), (512, 768))
plt.figure(4), plt.imshow(yr, 'gray', clim=(0, 255))

for i, j in enumerate([2.8, 0.9]):
    print(f'i: {i} j: {j}')
    Q2 = j
    Ybq = jl.bquant(Yb, jl.jpgqmtx()*Q2)
    Ybr = jl.brec(Ybq, jl.jpgqmtx()*Q2)
    yr = jl.ibdct(Ybr, (8, 8), (512, 768))
    plt.figure(i + 10), plt.imshow(yr, 'gray', clim=(0, 255))
    plt.title(f'reconstructed image Q2={j}')
    print("MEAN:", JPEGMEAN*Q2, "Ebpp:", 12/(JPEGMEAN*Q2))
    print(f'MSE: {mse(y, yr)}')
    print(f'PSNR: {psnr(y, yr)}')

plt.show(block=True)