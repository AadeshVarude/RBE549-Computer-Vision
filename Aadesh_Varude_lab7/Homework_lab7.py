import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np


def detect_circles(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
    gray = cv2.GaussianBlur(gray, (7, 7), 3) # Implementing gaussianc blurr
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0]/8, param1=100, param2=30, minRadius=1, maxRadius=100) # using houhg circles to detect circles in the image

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.3, 30, param1=150, param2=200, minRadius=0, maxRadius=0)
    
    # Drwaing the circle with pink colour circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1]) 
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
    return img,circles

def recognise_coins(img,circle):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to Grayscale
    sift = cv2.SIFT_create(2000) # Defining the sift creator
    # defining flann and it paramenters 
    index_params=dict(algorithm=1,trees=5)
    search_params=dict(checks=500)
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    
    # Defining the dollar sum and the coin count to 0
    dollar_sum=0
    penny_count=0
    qaurter_count=0
    dime_count=0
    nickel_count=0

    # getting coing templates from online images 
    penny=cv2.imread("Aadesh_Varude_lab7/templates/penny.jpeg") # book in the images
    penny=cv2.cvtColor(penny, cv2.COLOR_BGR2GRAY)

    quarter=cv2.imread("Aadesh_Varude_lab7/templates/quarter.jpeg") # book in the images
    quarter=cv2.cvtColor(quarter, cv2.COLOR_BGR2GRAY)

    dime=cv2.imread("Aadesh_Varude_lab7/templates/dime.jpeg") # book in the images
    dime=cv2.cvtColor(dime, cv2.COLOR_BGR2GRAY)

    nickle=cv2.imread("Aadesh_Varude_lab7/templates/nickel.jpeg") # book in the images
    nickle=cv2.cvtColor(nickle, cv2.COLOR_BGR2GRAY)



    # # Reading the coin template images capture from real coins 

    # penny=cv2.imread("Aadesh_Varude_lab7/templates/penny_cam.jpeg") # book in the images
    # penny=cv2.cvtColor(penny, cv2.COLOR_BGR2GRAY)

    # quarter=cv2.imread("Aadesh_Varude_lab7/templates/quarter_came.jpeg") # book in the images
    # quarter=cv2.cvtColor(quarter, cv2.COLOR_BGR2GRAY)

    # dime=cv2.imread("Aadesh_Varude_lab7/templates/dime.jpeg") # book in the images
    # dime=cv2.cvtColor(dime, cv2.COLOR_BGR2GRAY)

    # nickle=cv2.imread("Aadesh_Varude_lab7/templates/nickel_cam.jpeg") # book in the images
    # nickle=cv2.cvtColor(nickle, cv2.COLOR_BGR2GRAY)
    
    #Loop to manuver through all the circles detected in the image
    for c in circle:
        #Centre and the radius of the detected circle
        x,y,r=c
        roi=img[y - r : y + r, x - r : x + r] # Extracting the region of interest (Here I am taking on ly the area where the ircle is present) 
        
        # Getting the keypoints and descriptors of all the coins and images
        kp_penny,des_penny=sift.detectAndCompute(penny,None)
        kp_qaurter,des_quarter=sift.detectAndCompute(quarter,None)
        kp_dime,des_dime=sift.detectAndCompute(dime,None)
        kp_nickle,des_nickle=sift.detectAndCompute(nickle,None)
        
        kp_img,des_img=sift.detectAndCompute(roi,None)
        
        # Finding the all the matches in the image
        penny_matches=flann.knnMatch(des_img,des_penny,k=2)
        quarter_matches=flann.knnMatch(des_img,des_quarter,k=2)
        dime_matches=flann.knnMatch(des_img,des_dime,k=2)
        nickle_matches=flann.knnMatch(des_img,des_nickle,k=2)
        
        # adiing the good matches to the list
        penny_good_matches = [m for m, n in penny_matches if m.distance < 0.75 * n.distance]
        quarter_good_matches = [m for m, n in quarter_matches if m.distance < 0.75 * n.distance]
        dime_good_matches = [m for m, n in dime_matches if m.distance < 0.75 * n.distance]
        nickle_good_matches = [m for m, n in nickle_matches if m.distance < 0.75 * n.distance]


        # print(len(penny_good_matches),len(quarter_good_matches),len(nickle_good_matches))
        # Storing the lengths for all the mathes found
        lengths=[len(penny_good_matches),len(quarter_good_matches),len(nickle_good_matches)]
        lengths=np.array(lengths)
        # getiing indexes for maximum length 
        idx=np.argmax(lengths)
        
        #Counting the total mumber of coins and the total dollar sum
        if sum(lengths)==0:
            continue
        elif idx==0:
            penny_count+=1
            dollar_sum+=0.01
        elif idx==1:
            qaurter_count+=1
            dollar_sum+=0.25   
        elif len(dime_good_matches)>=10:
            dime_count+=1
            dollar_sum+=0.1
        elif idx==2:
            nickel_count+=1
            dollar_sum+=0.05
    # print("printing count",penny_count,nickel_count,dime_count,qaurter_count)
    return round(dollar_sum,2)


# Main code to capture the video form the webcam and proces the coing detection and sum
web_cam = cv2.VideoCapture(0) # reading feed from the webcamera 
while True:
    ret, image_frames = web_cam.read()
    
    if ret==False:
        print("webcam not capturing the video")
        break
    else:
        cv2.imshow('frame', image_frames)
        
        #Saving the key press in the variable key
        key=cv2.waitKey(1)
        # Press c to start the algoithm
        if key ==  ord("c"):
            print("in c")
            while True:
                # getting the image frames
                ret, image_frames = web_cam.read()
                img,circles=detect_circles(image_frames) # detecting the circles and obtaining the image with circles on it from the given image
                cv2.imshow('frame', img)
                # print("Image is captured")
                key=cv2.waitKey(1)
                if key== 27:
                    break
            # Obtaing the total coin cost in the image
            dollar=recognise_coins(img,circles[0])
            print(dollar)

            # setting fonts and location for the text to be displayed on the screen
            font = cv2.FONT_HERSHEY_SIMPLEX # Setting up the font
            color=(255,255,255)
            scale=1
            thickness=2
            # Showing the final image
            img = cv2.putText(img, str(dollar)+'$',(10,25),font,scale,color, thickness)
            cv2.imshow('frame', img)
            cv2.waitKey(0 )
            cv2.destroyAllWindows
        elif key==27:
            break
                







#--------------------------------------------------------------------------------------------------------------------------------------------#
#Part2
#Readung images and storing them in a list
img_fn = ["Aadesh_Varude_lab7/IMAGE_1.JPG", "Aadesh_Varude_lab7/IMAGE_2.JPG", "Aadesh_Varude_lab7/IMAGE_3.JPG"]
img_list = [cv2.imread(fn) for fn in img_fn]

#Defining the merge mertens function
merge_mertens = cv2.createMergeMertens()
#Implementing mertens algorithm
res_mertens = merge_mertens.process(img_list)
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
cv2.imshow("img", res_mertens_8bit)
cv2.imwrite("Aadesh_Varude_lab7/Results/Mertens_result.png", res_mertens_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows


#--------------------------Experiment section for captured image and coing recognisation ------------------------------------------------#

# og_img1 = cv2.imread('Aadesh_Varude_lab7/experiment_image/coins_prof.png', cv2.IMREAD_COLOR) #Loading Image
# og_img1=cv2.resize(og_img1,(800,600))
# img,circles=detect_circles(og_img1)
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows
# dollar=recognise_coins(og_img1,circles[0])

# print(dollar)
# font = cv2.FONT_HERSHEY_SIMPLEX # Setting up the font
# color=(255,255,255)
# scale=2
# thickness=2
# image_frames = cv2.putText(img, str(dollar)+'$',(10,50),font,scale,color, thickness)

# cv2.imshow("img", image_frames)
# cv2.imwrite('Aadesh_Varude_lab7/Results/coin_exp7.png',image_frames)
# cv2.waitKey(0)
# cv2.destroyAllWindows
