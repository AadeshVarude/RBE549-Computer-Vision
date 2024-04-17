import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Aadesh_Varude_Homework_Lab8/Images/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

    # Draw and display the corners
#     cv.drawChessboardCorners(img, (7,6), corners2, ret)
#     cv.imshow('img', img)
#     cv.waitKey(1000)
# cv.destroyAllWindows()

#-------------------------------Performing the camera calibration-------------------------------#
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Printing the camera matrix and the distortion matrix  and saving it in an nyp file
print("camera matrix : ", mtx )
print("distorition matrix : ",dist)
np.save('Aadesh_Varude_Homework_Lab8/Result/opencv_exp_camera_matrix',mtx)
np.save('Aadesh_Varude_Homework_Lab8/Result/opencv_exp_distortion_matrix',dist)



#-------------------------- To view the undistored image---------------------------------------------#

# img = cv.imread('Aadesh_Varude_Homework_Lab8/Images/left02.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('Aadesh_Varude_Homework_Lab8/Result/Opencv_calibresult_1.png', dst)

#-------------------------Calcualtion the mean reprojection error---------------------------------#
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) # reprojecting the points using the camera calibration parameters
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2) # Calculating the errors after reprojection
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )
np.save('Aadesh_Varude_Homework_Lab8/Result/opencv_exp_reprojection_error',np.array(mean_error))