import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# reading the left and right images
imgL = cv.imread('Aadesh_Varude_Lab9/Images_dataset/aloeL.jpg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('Aadesh_Varude_Lab9/Images_dataset/aloeR.jpg', cv.IMREAD_GRAYSCALE)
stereo = cv.StereoBM.create(numDisparities=128, blockSize=15) # Defining the number of disparties and blocksize
disparity = stereo.compute(imgL,imgR) # computing the disparity using the library

# Plotting and saving the disparity images
plt.imshow(disparity,'gray')
plt.imsave('Aadesh_Varude_Lab9/Results/disparity.jpg',disparity)
plt.show()