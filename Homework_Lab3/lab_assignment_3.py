import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np


def ash_sobel(gray,x,y,ddepth,kernel_size=3):
    gray=cv2.GaussianBlur(gray, (3,3), 0) # blurring the image for noise removal
    sobel_x_kernel=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # defining the operators
    sobel_y_kernel=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    if x:
        sobel=cv2.filter2D(gray,ddepth=ddepth,kernel=sobel_x_kernel) # convolving the operators
    elif y:
        sobel=cv2.filter2D(gray,ddepth=ddepth,kernel=sobel_y_kernel) # convolving the operators
    return sobel
def ash_laplacian(gray,ddepth,kernel_size=3):
    gray=cv2.GaussianBlur(gray, (3,3), 0) # blurring the image for noise removal
    laplacian_kernel=np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # defining the operators
    laplacian=cv2.filter2D(gray,ddepth=ddepth,kernel=laplacian_kernel)# convolving the operators
    return laplacian



web_cam = cv2.VideoCapture(0) # reading feed from the webcamera 
RED = [0,0,255]


vid_rec_flag=False
while True:
    ret, image_frames = web_cam.read()
    if ret==False:
        print("webcam not capturing the video")
        break
    else:
 
        cv2.imshow('frame', image_frames)
        white_image_frame=255*np.ones(image_frames.shape)
        #Saving the key press in the variable key
        key=cv2.waitKey(1)

        #Saving the image if key c is pressed
        if key ==  ord("c"):
            cv2.imwrite('Captured_Image.png',image_frames)
            #Adding a flash like feature by setting the values of all the image pixesl to white and adding a delay to get a flas feel
            cv2.imshow('frame',white_image_frame)
            cv2.waitKey(100)
            print("Image is captured")
        elif key ==  ord("v"):
            if not vid_rec_flag:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_video = cv2.VideoWriter('captured_video.avi', fourcc, 20.0, (640, 480))
                vid_rec_flag=True
                print("recording stared press v to stop")
            else:
                out_video.release()
                vid_rec_flag=False
                print("recorinding stopped")
        if vid_rec_flag:
            out_video.write(image_frames)
        elif key ==  ord("e"):
            print("Extrating red color from image/ press esp to exit")
            lower_red = np.array([160,50,50]) # lower bound for red color
            upper_red = np.array([180,255,255]) # uppper bound for red color
            while True:
                ret, image_frames = web_cam.read()
                hsv_image = cv2.cvtColor(image_frames, cv2.COLOR_BGR2HSV) # converting the RGB image to HSV format
                mask = cv2.inRange(hsv_image, lower_red, upper_red) # finding all the objects in the image for the red color range
                red_object_img=cv2.bitwise_and(image_frames, image_frames, mask=mask) # using a bitwise operator between image frames and mask
                cv2.namedWindow('Red Object Image') # created a new window to display the red object
                cv2.imshow('frame', red_object_img)
                cv2.imshow('Red Object Image', red_object_img)
                key=cv2.waitKey(1)
                # cv2.imshow('frame', image_frames)
                if key==27:
                    break
        elif key ==  ord("r"):
            rows,cols,_ = image_frames.shape
            print("Showing the rotated Image/ Press esc to stop the rotation")
            while True:
                ret, image_frames = web_cam.read()
                M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),10,1) # Getting the rotation matric for 10 degrees
                rotated_image= cv2.warpAffine(image_frames,M,(cols,rows)) # wraping the image frames with the rotation matrix
                cv2.namedWindow('Rotated Image') 
                cv2.imshow('Rotated Image', rotated_image)
                key=cv2.waitKey(1)
                # cv2.imshow('frame', image_frames)
                if key==27:
                    break
            # cv2.imwrite('Rotated_Captured_Image.png',image_frames)
        elif key==ord("b"):
            cv2.namedWindow('Blurred Image')
            # Initialize sigma values
            sigma_x = 5
            sigma_y = 5
            # Create trackbars for sigma values (x and y) 
            cv2.createTrackbar('Sigma x', 'Blurred Image', sigma_x, 30, lambda x: None)
            cv2.createTrackbar('Sigma y', 'Blurred Image', sigma_y, 30, lambda x: None)
            print("Showing the blurred Image/ Press esc to stop")
            while True:
                ret, image_frames = web_cam.read()
                # Get the current sigma values from the trackbars
                sigma_x = cv2.getTrackbarPos('Sigma x', 'Blurred Image')
                sigma_y = cv2.getTrackbarPos('Sigma y', 'Blurred Image')

                # Apply Gaussian blur to the image
                # Setting the kernel (0,0) as by doing this you calculate the kernel size using sigma_x and sigma_y values
                blurred_image = cv2.GaussianBlur(image_frames,(0, 0), sigma_x, sigma_y)

                # Display the blurred image
                cv2.imshow('Blurred Image', blurred_image)
                #Command for the esp pressed to exit the screen        
                key=cv2.waitKey(1)
                if key==27:
                    break
        elif key ==  ord("t"):
            print("Showing the rotated Image/ Press esc to stop the rotation")
            while True:
                ret, image_frames = web_cam.read()
                gray = cv2.cvtColor(image_frames, cv2.COLOR_BGR2GRAY) # COnverting the image frames to gray
                ret,thresholded_img = cv2.threshold(gray,110,255,cv2.THRESH_BINARY) # Thresholding that images using lower and higher bounds
                cv2.namedWindow('Thresholded Image')
                cv2.imshow('Thresholded Image', thresholded_img)
                key=cv2.waitKey(1)
                if key==27:
                    break

        elif key==ord("s"):
            key=cv2.waitKey(0)
            if key==ord("x"):
                cv2.namedWindow('Sobel x')
                kernel_size_x = 5
                # Create trackbars for kernel size values in x
                cv2.createTrackbar('Kernel Size x', 'Sobel x', kernel_size_x, 30, lambda x: None)
                
                print("doing sobel s_x")
                while True:
                    ret, image_frames = web_cam.read()
                    # Get the current sigma values from the trackbars
                    kernel_size_x = cv2.getTrackbarPos('Kernel Size x', 'Sobel x')
                    if kernel_size_x%2==0: #Making the kernel size odd
                        kernel_size_x+=1
                    
                    # converting the image to gray scale
                    gray = cv2.cvtColor(image_frames, cv2.COLOR_BGR2GRAY)
                    # Applying gaussian blur to remove noise
                    gray=cv2.GaussianBlur(gray, (3,3), 0)
                    # Applying sobel operator in x direction
                    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kernel_size_x)
                    # Display the sobel in x 
                    cv2.imshow('Sobel x', sobelx)
                    #Command for the esp pressed to exit the screen        
                    key=cv2.waitKey(1)
                    if key==27:
                        break
            if key==ord("y"):
                
                cv2.namedWindow('Sobel y')
                kernel_size_y = 5

                # Create trackbars for kernel size values in y
                cv2.createTrackbar('Kernel Size y', 'Sobel y', kernel_size_y, 30, lambda x: None)
                
                print("doing sobel s_y")
                while True:
                    ret, image_frames = web_cam.read()
                    # Get the current sigma values from the trackbars
                    kernel_size_y = cv2.getTrackbarPos('Kernel Size y', 'Sobel y')
                    if kernel_size_y%2==0:
                        kernel_size_y+=1
                    # converting the image to gray scale
                    gray = cv2.cvtColor(image_frames, cv2.COLOR_BGR2GRAY)
                    #applying gausssian blur to remove noise
                    gray=cv2.GaussianBlur(gray, (3,3), 0)
                    # Applying sobel operator in y direction 
                    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=kernel_size_y)
                    # Display the sobel in y 
                    cv2.imshow('Sobel y', sobely)
                    
                    #Command for the esp pressed to exit the screen        
                    key=cv2.waitKey(1)
                    if key==27:
                        break
        elif key==ord("d"):
            cv2.namedWindow('Canny Image')

            # Initialize x and y values
            x = 1
            y = 1

            # Create trackbars for  values x and y
            cv2.createTrackbar('x', 'Canny Image', x, 5000, lambda x: None)
            cv2.createTrackbar('y', 'Canny Image', y, 5000, lambda x: None)
            print("Showing the Canny Image/ Press esc to stop")
            while True:
                ret, image_frames = web_cam.read()
                # Get the current  values from the trackbars
                x = cv2.getTrackbarPos('x', 'Canny Image')
                y = cv2.getTrackbarPos('y', 'Canny Image')

                # Apply canny edge detectio to the image frames
                Canny_image = cv2.Canny(image_frames, x, y)

                # Display the Canny image
                cv2.imshow('Canny Image', Canny_image)
                #Command for the esp pressed to exit the screen        
                key=cv2.waitKey(1)
                if key==27:
                    break
        elif key == 52:
            cv2.namedWindow('gray')
            # cv2.namedWindow('sobel_y')
            # cv2.namedWindow('sobel_x')
            # cv2.namedWindow('laplacian')

            while True:
                    ret, image_frames = web_cam.read()
                    # converting the image to gray scale
                    gray = cv2.cvtColor(image_frames, cv2.COLOR_BGR2GRAY)
                    # displaying original stream of windows
                    cv2.imshow('gray', gray)
                    # Applying sobel operator in y direction 
                    sobely = ash_sobel(gray,0,1,cv2.CV_64F,3)
                    # Display the sobel in y
                    cv2.imshow('soble_y', sobely)
                    # # Applying sobel operator in x direction 
                    sobelx = ash_sobel(gray,1,0,cv2.CV_64F,3)
                    # # Display the sobel in x
                    cv2.imshow('sobel_x', sobelx)
                    # # Applying laplacian
                    laplacian = ash_laplacian(gray,cv2.CV_64F,3)
                    # # Display the sobel in x
                    cv2.imshow('laplacian', laplacian)
                    key=cv2.waitKey(1)
                    if key==27:
                        break
        
       
        elif key==27:
            break