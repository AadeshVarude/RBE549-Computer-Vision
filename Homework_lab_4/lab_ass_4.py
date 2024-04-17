import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np

sift = cv2.SIFT_create(nfeatures=500) # Defining the sift creator

# This function takes the gray and original image and calculates the harris corners
def cal_harris(gray_img,image):
    dst=cv2.cornerHarris(gray_img,2,3,0.04)
    dst = cv2.dilate(dst,None)
    image[dst>0.08*dst.max()]=[0,0,255] # setting threshold to get the orner in the image
    return image

# This function takes the gray and original image and calculates the Sift 
def cal_sift(gray_img,image):
    kp, _ = sift.detectAndCompute(image, None) # extracting key point 
    img=cv2.drawKeypoints(gray_img,kp,image,color=(0, 255, 0)) # drawing the key points on the image
    return img


image = cv2.imread('Homework_lab_4/UnityHall.png') #Loading Image
center_of_img=(image.shape[1]/2,image.shape[0]/2)

#Rotating the  image
M = cv2.getRotationMatrix2D((center_of_img),10,1) # Getting the rotation matric for 10 degrees
rotated_image= cv2.warpAffine(image,M,(image.shape[1],image.shape[0])) # wraping the image frames with the rotation matrix

#Up scaling teh image by 20 percent
up_scaled_image = cv2.resize(image, None, fx=1.2, fy=1.2,interpolation = cv2.INTER_CUBIC) # scale up by 20 %

#Down scaling teh image by 20 percent
down_scaled_image = cv2.resize(image, None, fx=0.8, fy=0.8) # scale down by 20 %

# Define three points to create an affine transformation
src_points = np.array( [[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1]] ).astype(np.float32)
dst_points = np.array( [[0, image.shape[1]*0.33], [image.shape[1]*0.85, image.shape[0]*0.25], [image.shape[1]*0.15, image.shape[0]*0.7]] ).astype(np.float32)
# Calculate the transformation matrix
matrix = cv2.getAffineTransform(src_points, dst_points)
# Apply the affine transformation
affine_image = cv2.warpAffine(image, matrix, image.shape[1::-1])

# Applying perspective transform to the image
pts1 = np.array( [[0, 0], [image.shape[1] - 1, 0],[0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1] ]).astype(np.float32)
pts2= np.array([[50, 50], [480, 50], [50, 250], [480, 480]], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(image,M,(500,500))


#-------------------------Code to display all the image--------------------#
cv2.imshow('img',image)
# cv2.imwrite('Homework_lab_4/Results/Orignal.png', image)
cv2.imshow('affine_image',affine_image)
# cv2.imwrite('Homework_lab_4/Results/affine_image.png', affine_image)
cv2.imshow('rotated_image',rotated_image)
# cv2.imwrite('Homework_lab_4/Results/rotated_image.png', rotated_image)
cv2.imshow('up_scaled_image',up_scaled_image)
# cv2.imwrite('Homework_lab_4/Results/up_scaled_image.png', up_scaled_image)
cv2.imshow('down_scaled_image',down_scaled_image)
# cv2.imwrite('Homework_lab_4/Results/down_scaled_image.png', down_scaled_image)
cv2.imshow('Perspective',dst)
# cv2.imwrite('Homework_lab_4/Results/Perspective.png', dst)

#-------------------------- Converting all images to gray------------------------------#

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray_upscale=cv2.cvtColor(up_scaled_image,cv2.COLOR_BGR2GRAY)
gray_downscale=cv2.cvtColor(down_scaled_image,cv2.COLOR_BGR2GRAY)
gray_affine=cv2.cvtColor(affine_image,cv2.COLOR_BGR2GRAY)
gray_dst=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
gray_rotated=cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)


# #--------------------------Calculating harris Corners----------------------------------#
harris_image=cal_harris(gray,image)
cv2.imshow('Harris_img',harris_image)
# cv2.imwrite('Homework_lab_4/Results/Orignal_harris_image.png', harris_image)

harris_up_scale_img=cal_harris(gray_upscale,up_scaled_image)
cv2.imshow('harris_up_scael_img',harris_up_scale_img)
# cv2.imwrite('Homework_lab_4/Results/harris_up_scale_img.png', harris_up_scale_img)

harris_down_scale_img=cal_harris(gray_downscale,down_scaled_image)
cv2.imshow('harris_down_scale_img',harris_down_scale_img)
# cv2.imwrite('Homework_lab_4/Results/harris_down_scale_img.png', harris_down_scale_img)

harris_affine_image=cal_harris(gray_affine,affine_image)
cv2.imshow('harris_affine_image',harris_affine_image)
# cv2.imwrite('Homework_lab_4/Results/harris_affine_image.png', harris_affine_image)

harris_dst_image=cal_harris(gray_dst,dst)
cv2.imshow('Perspective',harris_dst_image)
# cv2.imwrite('Homework_lab_4/Results/harris_perspective_image.png', harris_dst_image)

harris_rotated_image=cal_harris(gray_rotated,rotated_image)
cv2.imshow('harris_rotated_image',harris_rotated_image)
# cv2.imwrite('Homework_lab_4/Results/Orignal_harris_image.png', harris_image)

# #-------------------------------------------------------------------------#

# #--------------------------Calculating harris Corners----------------------------------#
sift_image=cal_sift(gray,image)
cv2.imshow('sift_img',sift_image)
# cv2.imwrite('Homework_lab_4/Results/Orignal_harris_image.png', sift_image)

sift_up_scale_img=cal_sift(gray_upscale,up_scaled_image)
cv2.imshow('sift_up_scael_img',sift_up_scale_img)
# cv2.imwrite('Homework_lab_4/Results/sift_up_scale_img.png', sift_up_scale_img)

sift_down_scale_img=cal_sift(gray_downscale,down_scaled_image)
cv2.imshow('sift_down_scale_img',sift_down_scale_img)
# cv2.imwrite('Homework_lab_4/Results/sift_down_scale_img.png', sift_down_scale_img)

sift_affine_image=cal_sift(gray_affine,affine_image)
cv2.imshow('sift_affine_image',sift_affine_image)
# cv2.imwrite('Homework_lab_4/Results/sift_affine_image.png', sift_affine_image)

sift_dst_image=cal_sift(gray_dst,dst)
cv2.imshow('Perspective',sift_dst_image)
# cv2.imwrite('Homework_lab_4/Results/sift_perspective_image.png', sift_dst_image)

sift_rotated_image=cal_sift(gray_rotated,rotated_image)
cv2.imshow('sift_rotated_image',sift_rotated_image)
# cv2.imwrite('Homework_lab_4/Results/sift_rotated_image.png', sift_rotated_image)
# #-------------------------------------------------------------------------#




cv2.waitKey(0)
cv2.destroyAllWindows()