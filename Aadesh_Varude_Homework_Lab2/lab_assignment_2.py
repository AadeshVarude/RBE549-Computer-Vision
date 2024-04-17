import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np

web_cam = cv2.VideoCapture(0) # reading feed from the webcamera 
RED = [0,0,255]

vid_rec_flag=False
while True:
    ret, image_frames = web_cam.read()
    
    if ret==False:
        print("webcam not capturing the video")
        break
    else:
      # Get current date and time  
        date_time = str(datetime.datetime.now())
        #Fixing the origin to insert text at the bottom right
        x_cor=image_frames.shape[0]-100
        y_cor=image_frames.shape[1]-200
        
      # write the date time in the video frame images
        font = cv2.FONT_HERSHEY_SIMPLEX # Setting up the font
        color=(255, 255, 255)
        scale=0.5
        thickness=1
        image_frames = cv2.putText(image_frames, date_time,(x_cor,y_cor),font,scale,color, thickness)

        text_size,_ = cv2.getTextSize(date_time, font, scale,thickness)
        x_size, y_size=text_size
        
        #shifting the roi of time to top right
        roi=image_frames[y_cor-y_size:y_cor,x_cor:x_cor+x_size] 
        image_frames[30-y_size:30,380:380+x_size]=roi
        
        # Reading the opencv image and then putting selecting the roi and adding the opencv logo to top right corner
        opencv_logo=cv2.imread("opencv_logo.png")
        roi_for_opencv_logo=image_frames[0:opencv_logo.shape[0],0:opencv_logo.shape[1]]
        #adding the image logo to the roi
        dis=cv2.addWeighted(roi_for_opencv_logo,0.6,opencv_logo,0.4,0)
        #Changin the image frames with the blended logo
        image_frames[0:opencv_logo.shape[0],0:opencv_logo.shape[1]] = dis
        
        #Making a red border on the video frmaes using the cv2 function
        image_frames= cv2.copyMakeBorder(image_frames,10,10,10,10,cv2.BORDER_CONSTANT,value=RED)
        
        cv2.imshow('frame', image_frames)

        #Creating a flash like image
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

        # Saving the videos if key v is pressed
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
        elif key== ord("s"):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # using this kernel to sharpen the images in the video stream
            
            while True:
                ret, image_frames = web_cam.read()
                # Sharpen the image
                sharpened_img = cv2.filter2D(image_frames, -1, kernel) # convolving the kernel with the iamge frames to get the sharpened iamge
                cv2.namedWindow('Sharpened Image')
                cv2.imshow('Sharpened Image', sharpened_img)
                key=cv2.waitKey(1)
                # cv2.imshow('frame', image_frames)
                if key==27:
                    break

        elif key==27:
            break

web_cam.release()
cv2.destroyAllWindows()



