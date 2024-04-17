import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cv2
clicked_points=[]
# Function to draw the lines
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2



# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
   
   # checking for left mouse clicks
   if event == cv2.EVENT_LBUTTONDOWN:
      clicked_points[-1].append((x, y))
      print('Left Click')
      print(f'({x},{y})')
      cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# read the input image
img = cv.imread('Aadesh_Varude_Lab9/Images_dataset/globe_center.jpg', cv.IMREAD_GRAYSCALE)  #queryimage # left image

clicked_points.append([])
# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
   cv2.imshow('Point Coordinates', img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()
# Reading the images
img = cv.imread('Aadesh_Varude_Lab9/Images_dataset/globe_right.jpg', cv.IMREAD_GRAYSCALE)  #queryimage # left image

clicked_points.append([])
# create a window
cv2.namedWindow('Point Coordinates')

# bind the callback function to window
cv2.setMouseCallback('Point Coordinates', click_event)

# display the image
while True:
   cv2.imshow('Point Coordinates', img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break
cv2.destroyAllWindows()

# clicked_points=[[(146, 90), (167, 585), (385, 500), (389, 125), (402, 128), (399, 495), (529, 444), (543, 149), (550, 150), (538, 441), (625, 406)], [(31, 155), (46, 431), (226, 429), (221, 158), (235, 157), (242, 428), (417, 426), (417, 156), (434, 158), (431, 426), (607, 424)]]
pts1=clicked_points[0]
pts2=clicked_points[1]
print('pts1',pts1)
print('pts2',pts2)
# pts1 =[(32, 156), (45, 431), (220, 154), (226, 428), (435, 160), (430, 426), (619, 158), (605, 423)]
# pts2 =[(95, 183), (97, 405), (181, 174), (184, 434), (340, 153), (331, 482), (563, 124), (539, 554)]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# #Finding the fundamental matrix
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_8POINT)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#Printing fundamental matrix and finding the rank 
print(F)
print("Rank",np.linalg.matrix_rank(F))


img1 = cv.imread('Aadesh_Varude_Lab9/Images_dataset/globe_center.jpg', cv.IMREAD_GRAYSCALE)  #queryimage # left image
img2 = cv.imread('Aadesh_Varude_Lab9/Images_dataset/globe_right.jpg', cv.IMREAD_GRAYSCALE) #trainimage # right image

# # Finding and plotting lines and finding the epipole

lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()