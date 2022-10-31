#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# subject: Image Processing
# name: maryam farshchian
# 


# loading nessery packages
import os
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import matplotlib.image as img
from scipy import fftpack
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cmath


# In[ ]:


###
### PART 1-1 and 1-3
###

im = np.mean(cv2.imread("c:/b3.jpg"), axis=2) / 255
print('height and width:',im.shape[0],' , ',im.shape[1])


# In[15]:



#img = cv2.imread('C:/b3.jpg',0)
img = cv2.imread('c:/b3.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
f2 = np.fft.fft2(img)
plt.imshow(abs(f2))
plt.title('DFT')

im = Image.fromarray(abs(f2))
plt.imshow(im)
#im.show()
plt.imsave('c:/b3-DFT.png',im)


# In[20]:



from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

image = io.imread('c:/b3.jpg')
ax = plt.hist(image.ravel(), bins = 256)
plt.show()


# In[9]:



import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('c:/b3.jpg',0)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
#img = cv.imread('c:/b3.jpg',0)
equ = cv.equalizeHist(img)
#res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('c:/s2.png',equ)


# In[ ]:


###
### PART 2-1 and 2-2
###


# In[10]:


def median_filter(data, kernel_size):
    temp = []
    #The real floor division operator is “//”. 
    #It returns floor value for both integer and floating point arguments.
    indexer = kernel_size//2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
    
    for i  in range(len(data)):
        for j in range(len(data[0])):
            for z in range(kernel_size):
                
                if i+z-indexer < 0 or i+z-indexer > len(data)-1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j+z-indexer < 0 or j+indexer > len(data[0])-1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(data[i+z-indexer][j+k-indexer])
            temp.sort()
            data_final[i][j]=temp[len(temp) // 2]
            temp=[]
    
    return data_final     
                        
    


# In[11]:




noisy_img = cv2.imread('c:/d10.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
# apply median filter
filter_img = median_filter(noisy_img, 3)
plt.imsave('C:/d10Window3.png', filter_img, cmap='gray')


# In[12]:




noisy_img = cv2.imread('c:/d70.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
# apply median filter
filter_img = median_filter(noisy_img, 3)
plt.imsave('C:/d70Window3.png', filter_img, cmap='gray')


# In[13]:




noisy_img = cv2.imread('c:/d70.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
# apply median filter
filter_img = median_filter(noisy_img, 5)
plt.imsave('C:/d70Window5.png', filter_img, cmap='gray')


# In[14]:




noisy_img = cv2.imread('c:/d70.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
# apply median filter
filter_img = median_filter(noisy_img, 7)
plt.imsave('C:/d70Window7.png', filter_img, cmap='gray')


# In[21]:




noisy_img = cv2.imread('c:/d70.jpg', cv2.IMREAD_GRAYSCALE).astype(float)
# apply median filter
filter_img = median_filter(noisy_img, 9)
plt.imsave('C:/d70Window9.png', filter_img, cmap='gray')

