import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np

sift = cv2.SIFT_create() # Defining the sift creator

# Funciton two stiching the images on the canvas
def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0) # taking the max element indexs
    ind1 = np.where(indx ==0) # storing the indices where the value is 0
    ind2 = np.where(indx==1) # storing the indices where the value is 1
    img = np.zeros_like(img1) # creating a canvas like image of zeros
    img[ind1] = img1[ind1] # stiching the two images on the canvas like image
    img[ind2] = img2[ind2]
    return  img

# this is a cropping function as jus to crop the final panorama to make it look good
def crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


# This function is a similar implementation of the lab work just the way of representing the image in the canvas is bit changed
def stich_panorama(images):
    sift = cv2.SIFT_create() # Defining the sift creator
    src_img=images[0]
    canvas = np.zeros((src_img.shape[0],src_img.shape[1]*len(images)+1,src_img.shape[2]),np.uint8)
    canvas[0:src_img.shape[0],0:0+ src_img.shape[1]] = src_img
    temp_img = canvas.copy()
    
    for i in range(len(images)-1):
        src_img=temp_img
        dest_img=images[i+1]
        gray_img1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray_img2 = cv2.cvtColor(dest_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        kp_src_img,des_src_img=sift.detectAndCompute(gray_img1,None)
        kp_dest_img, des_dest_img = sift.detectAndCompute(gray_img2, None)
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
        
        params=dict(matchColor = (0,255,0),singlePointColor = (255,0,0))
        sift_flann_img=cv2.drawMatches(src_img,kp_src_img,dest_img,kp_dest_img,good_matches,None,**params)
        
        src_pts = np.float32([kp_src_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dest_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)
        temp_img = cv2.warpPerspective(dest_img,H,(canvas.shape[1],canvas.shape[0]))
        canvas = image_stiching(canvas,temp_img)   
        
    return crop(canvas)






# # Deinfing image and extracting the sift feature points

# img1=cv2.imread("Homework_lab_5/lakshmila.jpg",cv2.IMREAD_GRAYSCALE) # book in the images
# kp_img,des_img=sift.detectAndCompute(img1,None)
# index_params=dict(algorithm=0,trees=5)
# search_params=dict()
# flann=cv2.FlannBasedMatcher(index_params,search_params)



# This function takes the gray and original image and calculates the harris corners
def cal_harris(gray_img,image):
    dst=cv2.cornerHarris(gray_img,2,3,0.04)
    dst = cv2.dilate(dst,None)
    image[dst>0.05*dst.max()]=[0,0,255] # setting threshold to get the orner in the image
    return image

# This function takes the gray and original image and calculates the Sift 
def cal_sift(gray_img,image):
    kp, _ = sift.detectAndCompute(image, None) # extracting key point 
    img=cv2.drawKeypoints(gray_img,kp,image,color=(0, 255, 0)) # drawing the key points on the image
    return img


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


        elif key ==ord("f"):
            cv2.namedWindow('Harris_corner')
            cv2.namedWindow('Sift')

            while True:
                    ret, image_frames = web_cam.read()
                    # converting the image to gray scale
                    gray = cv2.cvtColor(image_frames, cv2.COLOR_BGR2GRAY)
                    # getting harris coreners using the function
                    harris_image=cal_harris(gray,image_frames)
                    cv2.imshow('Harris_corner', harris_image)
                    #getting sift feature using the funtion
                    sift_image=cal_sift(gray,image_frames)
                    cv2.imshow('Sift', sift_image)
                    key=cv2.waitKey(1)
                    if key==27:
                        break
        # elif key ==ord("u"): # key for object detection
        #     while True:
        #         ret, image_frames = web_cam.read()
        #         cv2.namedWindow('Obj_detection')
        #         grayvideo=cv2.cvtColor(image_frames,cv2.COLOR_BGR2GRAY)
        #         kp_video, des_video = sift.detectAndCompute(grayvideo, None)
        #         matches=flann.knnMatch(des_img,des_video,k=2)
        #         good=[]
        #         for m,n in matches:
        #             if m.distance<n.distance*0.6:
        #                 good.append(m)
        #         match_img=cv2.drawMatches(img1,kp_img,grayvideo,kp_video,good,grayvideo)
        #         cv2.imshow('Object detected',match_img)
        #         key=cv2.waitKey(1)
        #         if key==27:
        #             break
        elif key ==ord("l"): #Key for performing panorama stiching
            images=[]
            # print('press w to store images')
            i=0
            while True:
                ret, image_frames = web_cam.read()
                # print("press c to capture images")
                cv2.imshow('img', image_frames)
                key=cv2.waitKey(1)
                if key==ord("w"):
                    images.append(image_frames)
                    cv2.imwrite("Aadesh_varude_lab6/image"+str(i)+'.png',image_frames) 
                    print("images appended")
                    i=i+1
                elif key==27:
                    break
            
            # print()
            panorama=stich_panorama(images)
            cv2.imshow('Panorama',panorama)
            cv2.imwrite("Aadesh_varude_lab6/panorama.png",panorama) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()

       
        elif key==27:
            break