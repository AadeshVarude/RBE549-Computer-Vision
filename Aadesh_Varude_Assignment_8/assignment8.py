import numpy as np
import cv2 
import math

# Reading the Image
img1 = cv2.imread('Assignment_8/texas.png') #Loading Image

#-----------------------Canny Edge detection algorithm ------------------------------------------------------#
img_1= cv2.GaussianBlur(img1,(5,5),0) # Perfoming gaussian blur to get clean edges
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale

canny_img = cv2.Canny(gray_img1, 150, 200, None, 3) # Performing Canny edge detection 

cv2.imshow('img', img1)
cv2.imshow('Canny', canny_img)
cv2.imwrite('Assignment_8/Results/Canny.png',canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------#
#------------------------------------------Hough Transform method to detect lines-----------------------------------------------------------------------#
cdst = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR) #Conversitng the canny image to gray
lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 200, None, 0, 0) # Detecting line using hough transfor in the canny image


A=[] # cos theta and sine theta
B=[] # rho values

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0] # getting the rho
        theta = lines[i][0][1] # getting the theta values
        if(1.4<theta<1.6): # Rejecting the theta values that are horizontal
            continue
        # Getting the cosine and sine values
        a = math.cos(theta) 
        b = math.sin(theta)
        A.append([a,b]) # Creating the A
        B.append(rho) # Creating the B
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA) # Drawing the line for visualization
cv2.imshow('lines', cdst)
cv2.imwrite('Assignment_8/Results/Lines.png',cdst)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------#
#---------------------SVD decomposition to solve for the t-----------------------------------------------#
# U, S, Vt = np.linalg.svd(A, full_matrices=False)
# V = Vt.T
# t = V @ np.linalg.inv(np.diag(S)) @ U.T @ B

# u, v = t
#----------------------------------------------------------------------------------------------------------------------#

#--------------Using the formula for calculating t----------------------------------------------------------------# 
A=np.array(A)
B=np.array(B)

t=np.linalg.inv(A.T@A)@A.T@B # Implementing the formulae
u, v = t
#---------------------------------------------------------------------------------------------------------------------#

# Mark the vanishing point with a red circle
result = img1.copy()
cv2.circle(result, (int(u), int(v)), 15, (0, 0, 255), -1)
cv2.imshow('img', result)
cv2.imwrite('Assignment_8/Results/Final_result.png',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------#