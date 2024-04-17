import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np

og_img1 = cv2.imread('Aadesh_varude_lab6/boston1.jpeg') #Loading Image
og_img2 = cv2.imread('Aadesh_varude_lab6/boston2.jpeg') #Loading Image

gray_img1 = cv2.cvtColor(og_img1, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
gray_img2 = cv2.cvtColor(og_img2, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale

#----------------SIFT FLANN BASED FEATURE MATHCING-------------------------#
sift = cv2.SIFT_create() # Defining the sift creator

# generating sift descriotors and keypoints
kp_src_img,des_src_img=sift.detectAndCompute(gray_img1,None)
kp_dest_img, des_dest_img = sift.detectAndCompute(gray_img2, None)

#-------------------------------------------- Flann Based Matching apporach ---------------------------------------------#
# These are the recommended paprameters to be passed for SIFT 
index_params=dict(algorithm=1,trees=5)
search_params=dict(checks=50)
# defining flann
flann=cv2.FlannBasedMatcher(index_params,search_params)
matches=flann.knnMatch(des_src_img,des_dest_img,k=2) # Finding the all the matches in the image
good_matches=[] # deifng a list to store good matches

# Iterating and thresholding and obtaing the best matches between the two images
for m,n in matches:
    if m.distance<n.distance*0.7:
        good_matches.append(m)

# Drawing and Visualiseing the matches 
#params for the drwaing colors for the matches
params=dict(matchColor = (0,255,0),
 singlePointColor = (255,0,0))

sift_flann_img=cv2.drawMatches(og_img1,kp_src_img,og_img2,kp_dest_img,good_matches,None,**params)
# cv2.imshow("SIFT AND FLANN MATHING",sift_flann_img)
# # cv2.imwrite('Aadesh_varude_lab6/Matched_features.png',sift_flann_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Get corresponding points in both images
src_pts = np.float32([kp_src_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_dest_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Finding Homography using those matches and using ransaca to reject the outliers
H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)

width = og_img1.shape[1] + og_img2.shape[1]

# Warping the second image with respect to the first 
panorama = cv2.warpPerspective(og_img2, H,  (width,og_img2.shape[0]))
# Laying over the image on the first one to stich the parameter
panorama[0:og_img1.shape[0], 0:og_img1.shape[1]] = og_img1

# Outputing the image
cv2.imshow("Panoram",panorama)
# cv2.imwrite('Aadesh_varude_lab6/Panorama.png',panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
