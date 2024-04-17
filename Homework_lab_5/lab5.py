import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np

book_img = cv2.imread('Homework_lab_5/book.jpg',cv2.IMREAD_GRAYSCALE) #Loading Image
table_img = cv2.imread('Homework_lab_5/table.jpg',cv2.IMREAD_GRAYSCALE) #Loading Image


#----------------SIFT FLANN BASED FEATURE MATHCING-------------------------#
sift = cv2.SIFT_create() # Defining the sift creator
# generating sift descriotors and keypoints
kp_src_img,des_src_img=sift.detectAndCompute(book_img,None)
kp_dest_img, des_dest_img = sift.detectAndCompute(table_img, None)

# visualizing the key points
img1=cv2.drawKeypoints(book_img,kp_src_img,book_img,color=(0, 255, 0))
img2=cv2.drawKeypoints(table_img,kp_dest_img,table_img,color=(0, 255, 0))

cv2.imshow('book_img',img1)
cv2.imshow('table_img',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Flann Based Matching apporach
# These are the recommended paprameters to be passed for SIFT and SURF algorithms
index_params=dict(algorithm=1,trees=5)
search_params=dict()
# defining flann
flann=cv2.FlannBasedMatcher(index_params,search_params)
matches=flann.knnMatch(des_src_img,des_dest_img,k=2) # Finding the all the matches in the image
good_matches=[] # deifng a list to store good matches

# Iterating and thresholding and obtaing the best matches between the two images
for m,n in matches:
    if m.distance<n.distance*0.6:
        good_matches.append(m)

#params for the drwaing colors for the matches
params=dict(matchColor = (0,255,0),
 singlePointColor = (255,0,0))

# Drawing the matches
sift_flann_img=cv2.drawMatches(book_img,kp_src_img,table_img,kp_dest_img,good_matches,None,**params)
cv2.imshow("SIFT AND FLANN MATHING",sift_flann_img)
cv2.imwrite('Homework_lab_5/Results/SIFT AND FLANN MATHING.png', sift_flann_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------------#
#-----------------------SIFT AND BRUTE FORCEMATCHING-------------------------------------------------------#
# BFMatcher with default params
bf = cv2.BFMatcher()
matches_1 = bf.knnMatch(des_src_img,des_dest_img,k=2)

good_bf_matches = []
# Thresholding and filters the bes matches 
for m,n in matches_1:
 if m.distance < 0.65*n.distance:
    good_bf_matches.append([m])

# Drwaing the mathces
bf_sift_mathced_img = cv2.drawMatchesKnn(book_img,kp_src_img,table_img,kp_dest_img,good_bf_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,**params)
cv2.imshow("SIFT AND BRUTE FORCE",bf_sift_mathced_img)
cv2.imwrite('Homework_lab_5/Results/SIFT AND BRUTE FORCE.png', bf_sift_mathced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------------------------------------------------------------------------------------------------------------------------------------------------#

#----------------SURF FLANN BASED FEATURE MATHCING----------------------------------------------------------------------#
# Defining the surf
surf =cv2.xfeatures2d.SURF_create(400) 
# Extracting key points and descriptors for the source and the normal image
kp_surf_src_img,des_surf_src_img=surf.detectAndCompute(book_img,None)
kp_surf_dest_img, des_surf_dest_img = surf.detectAndCompute(table_img, None)


img3=cv2.drawKeypoints(book_img,kp_surf_src_img,book_img,color=(255, 255, 0))
img4=cv2.drawKeypoints(table_img,kp_surf_dest_img,table_img,color=(255, 255, 0))

cv2.imshow('surf_book_img',img3)
cv2.imshow('surf_table_img',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Flann Based matching same parametere defination as above
index_params=dict(algorithm=1,trees=5)
search_params=dict()
flann=cv2.FlannBasedMatcher(index_params,search_params)
matches_2=flann.knnMatch(des_surf_src_img,des_surf_dest_img,k=2)
surf_good_matches=[]
# Thresholding and filters the best matches 
for m,n in matches_2:
    if m.distance<n.distance*0.6:
        surf_good_matches.append(m)

params=dict(matchColor = (0,255,0),
 singlePointColor = (255,0,0))

surf_flann_img=cv2.drawMatches(book_img,kp_surf_src_img,table_img,kp_surf_dest_img,surf_good_matches,None,**params)
cv2.imshow("SURF AND FLANN MATHING",surf_flann_img)
cv2.imwrite('Homework_lab_5/Results/SURF AND FLANN MATHING.png', surf_flann_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------------------------------#

#-----------------------SURF AND BRUTE FORCEMATCHING-----------------------------------------------------------------------------#

# BFMatcher with default params
bf = cv2.BFMatcher()
matches_3 = bf.knnMatch(des_surf_src_img,des_surf_dest_img,k=2)
# Thresholding and filters the bes matches 
good_bf_surf_matches = []
for m,n in matches_3:
 if m.distance < 0.65*n.distance:
    good_bf_surf_matches.append([m])

bf_surf_mathced_img = cv2.drawMatchesKnn(book_img,kp_surf_src_img,table_img,kp_surf_dest_img,good_bf_surf_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,**params)
cv2.imshow("SURF AND BRUTE FORCE",bf_surf_mathced_img)
cv2.imwrite('Homework_lab_5/Results/SURF AND BRUTE FORCE.png', bf_surf_mathced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-------------------------------------------------------------------------------------------------------------------------------#