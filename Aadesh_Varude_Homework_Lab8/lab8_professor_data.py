import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points
objp = np.zeros((11*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:11].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Aadesh_Varude_Homework_Lab8/Images/calibration_data/calibration_data/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,11), None) # Finding the corners with 7,11 pattern
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) # Refining the detected corners 
        imgpoints.append(corners2)
        
    # Draw and display the corners
#     cv.drawChessboardCorners(img, (7,11), corners2, ret)
#     cv.imshow('img', img)
#     cv.waitKey(1000)
# cv.destroyAllWindows()

# #-------------------------------Performing the camera calibration-------------------------------#
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Printing the camera matrix and the distortion matrix  and saving it in an nyp file
print("camera matrix : ", mtx )
print("distorition matrix : ",dist)
np.save('Aadesh_Varude_Homework_Lab8/Result/assigemnt_exp_camera_matrix',mtx)
np.save('Aadesh_Varude_Homework_Lab8/Result/assigemnt_exp_distortion_matrix',dist)



# #-------------------------- To view the undistored image---------------------------------------------#

# img = cv.imread('Aadesh_Varude_Homework_Lab8/Images/calibration_data/calibration_data/IMG_6502.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('Aadesh_Varude_Homework_Lab8/Result/prof_calibresult_1.png', dst)

# #-------------------------Calcualtion the mean reprojection error---------------------------------#
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) # reprojecting the points using the camera calibration parameters
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2) # Calculating the errors after reprojection
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
np.save('Aadesh_Varude_Homework_Lab8/Result/assignment_exp_reprojection_error',np.array(mean_error))