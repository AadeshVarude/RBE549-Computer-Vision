import cv2 # Importing necessary libraries
from cv2 import *
import datetime

web_cam = cv2.VideoCapture(0) # reading feed from the webcamera 

vid_rec_flag=False  # setting a flag to start and stop the video recording
# Create a window
cv2.namedWindow('frame')

# Create a trackbar
# cv2.createTrackbar('Zoom', 'frame', 1, 10, lambda x: None)


while True:
    ret, image_frames = web_cam.read()
    
    if ret==False:
        print("webcam not capturing the video")
        break
    else:
#         scale = cv2.getTrackbarPos('Zoom', 'frame')
#         #-----------------------BONUS POINT PART-------------------------------------------------#
#         #Get the frame dimensions
#         height, width, _ = image_frames.shape

#         #Cropping the image as pper th scale of zoom
#         center_x,center_y=int(height/2),int(width/2)

#         dist_x,dist_y= int(height/(scale*2)),int(width/(scale*2))

#         #Calculating the distance from the centre to the image as croppped as per the scale and distance calculated
#         min_x,max_x=center_x-dist_x,center_x+dist_x
#         min_y,max_y=center_y-dist_y,center_y+dist_y
#         # print("cropping the images",scale)

#         cropped = image_frames[min_x:max_x, min_y:max_y]
        
#         image_frames = cv2.resize(cropped, (width, height)) 

#         # cv2.imshow('frame', image_frames)
#         #-----------------------BONUS POINT PART END-------------------------------------------------#


# #---------------------------------------------------PART ONE FOR CAPTURING IMAGE AND RECONRDING VIDEO-------------------------#
#       # Get current date and time  
#         date_time = str(datetime.datetime.now())

#         #Fixing the origin to insert text at the bottom right
#         x_cor=image_frames.shape[0]-100
#         y_cor=image_frames.shape[1]-200
#         # print(x_cor)
#         # print(y_cor)

#       # write the date time in the video frame images
#         font = cv2.FONT_HERSHEY_SIMPLEX  # setting font 
#         image_frames = cv2.putText(image_frames, date_time,(x_cor,y_cor),font, 0.5,(255, 255, 255), 1) # defining put text to put the date and time at bottom right
        cv2.imshow('frame', image_frames)
        
        #Saving the key press in the variable key
        key=cv2.waitKey(1)

        #Saving the image if key c is pressed
        if key ==  ord("c"):
            #writing and storing the image
            cv2.imwrite('Coins.png',image_frames)
            print("Image is captured")

        # Saving the videos if key v is pressed
        elif key ==  ord("v"):
            if not vid_rec_flag: # recording the video by checking the flag
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_video = cv2.VideoWriter('captured_video.avi', fourcc, 20.0, (640, 480)) #Using video writer to capture video
                vid_rec_flag=True 
                print("recording stared press v to stop")
            else: # if the flag indicates recording to be stopped.
                out_video.release()
                vid_rec_flag=False
                print("recorinding stopped")
        if vid_rec_flag:
            out_video.write(image_frames)

        #Command for the esp pressed to exit the screen        
        elif key==27:
            break

web_cam.release()
cv2.destroyAllWindows()



