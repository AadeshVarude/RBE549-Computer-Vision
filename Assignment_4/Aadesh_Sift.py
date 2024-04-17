import cv2 # Importing necessary libraries
from cv2 import *
import datetime
import numpy as np
from scipy.signal import convolve2d as convolve
import math as m 
import matplotlib.pyplot as plt



row=256
col=256
img=cv2.imread('Assignment_4/lenna.png')
origin=img
print(img.shape)
#Image pre processing resizing converting to gray and normalizeing the image
img=cv2.resize(img,(row,col))
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # COnverting the image to gray
img=cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)




# Scale Space extrema Detection

sigma0=np.sqrt(2)
octave=3
level=3
# creatind a canvas of zeros and creating a Dog layered matrix
D = [np.zeros((int(row * 2**(2-i)) + 2, int(col * 2**(2-i)) + 2, level)) for i in range(1,octave+1)]

# Creating a temporary image by interpolation and making it of size 512 and then making border of one pixel all around thus size size becomes 514
temp_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
temp_img = cv2.copyMakeBorder(temp_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
num=0
#Iterating thorugh all octaves and levels to create a Dog stack of images
for i in range(1,octave+1):
    temp_D=D[i-1]
    for j in range(1,level+1):
        # Creating a Gaussian filter  by defing parameters given as per David Lowes matlab code
        scale=sigma0*(np.sqrt(2)**(1/level))** ((i-1) * level + j)
        p=level**(i-1)
        kernel_size=int(np.floor(6*scale))
        f = cv2.getGaussianKernel(kernel_size, scale) # it generates the kernel for gaussian filter 
        # now doing the difference ofgaussian by applying the filter and doing the pyramid of the various ocatves and levels
        L1=temp_img
        if(i==1 and j==1):
            L2=convolve(temp_img, f.reshape(1, kernel_size),mode='same') #Convolving to apply gaussian filter
            L2=convolve(L2, f.reshape(1, kernel_size),mode='same') #Convolving to apply gaussian filter
            temp_D[:,:,j-1]=L2-L1
            L1=L2
        else:
            L2=convolve(temp_img, f.reshape(1, kernel_size),mode='same')
            L2=convolve(L2, f.reshape(1, kernel_size),mode='same')
            temp_D[:,:,j-1]=L2-L1
            L1=L2
            if(j==level):
                temp_img=L1[1:-2,1:-2]
        
        # Viewing the dog pyramid results
        num+=1
        plot_img=255 * temp_D[:,:,j-1]
        
        # cv2.imshow('img'+str(num),plot_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
    # Check the sizes here
    D[i-1]=temp_D
    temp_img=temp_img[::2,::2]
    temp_img = cv2.copyMakeBorder(temp_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)


# Keypoint localization 
# Now that we have DOG then we find the extreme points
# Setting up the parameteres are per the David Lowes Code
interval=level-1  
number=0
flag=0
for i in range(2,octave+2):
    number=number+(2**(i-octave)*col)*(2*row)*interval
#creating an array to store the extrema locations
extrema=np.zeros(int(4*number))

for i in range(1,octave+1):
    m,n,_=D[i-1].shape
    m=m-2
    n=n-2
    volume=int(m*n/(4**(i-1))) # This is the search space for the each octave
    # print(volume)
    for k in range(2,interval+1):
        for j in range(1,volume+1):
        # Creating the x adn y location of the matrix given the row*col of the matrix:
            x=((j-1)/n)+1 
            x=int(x)
            y=np.remainder(j-1,m)+1
            #extracting the 27 values form 3 levels in an ocatave for a pixel neighbourhood
            sub=D[i-1][x:x+3,y:y+3,k-2:k+1]
            #calculating the maxima and the minima
            large=np.max(sub)
            little=np.min(sub)
            #storing the extrema octave, level, location in volume and the value that whether an maxima or minima
            if(large==D[i-1][x,y,k-1]):
                temp=np.array([i,k-1,j,1])
                extrema[flag:flag+4]=temp
                flag=flag+4 # see
            if(little==D[i-1][x,y,k-1]):
                temp=np.array([i,k-1,j,-1])
                extrema[flag:flag+4]=temp
                flag=flag+4 # see

# Extracting only extrema and rejecting all others
extrema = extrema[extrema != 0]

img_height, img_width = img.shape
extrema_values = extrema[2::4] # row col value
extrema_octaves = extrema[0::4] # octave value


# just reconstrcuting the x and y from the given vloume location and octave 
x = np.floor((extrema_values - 1) / ((img_width) / (2 ** (extrema_octaves-2)))) +1
y = np.remainder((extrema_values - 1), ((img_height) / (2 ** (extrema_octaves-2))))+1

ry = y / (2 ** (octave - 1 - extrema_octaves))
rx = x / (2 ** (octave - 1 - extrema_octaves))

#PLotting the extremas found
plt.figure(1)
plt.imshow(img,cmap='gray')
plt.scatter(ry,rx,marker='+',color='green')
plt.show()

# Accurate key point localisation

thershold=0.1 # as defined by the lowes code
r=10
extrema_volume=len(extrema)/4

# # print(extrema_volume)
m,n=img.shape
#second order kernel for x and y
secondorder_x=convolve([[-1,1],[-1,1]],[[-1,1],[-1,1]])            
secondorder_y=convolve([[-1,-1],[1,1]],[[-1,-1],[1,1]])            

# for all ocataves and levels convolving the dog images
for i in range(1,octave+1):
    for j in range(1,level+1):
        test=D[i-1][:,:,j-1]
        temp=-1/convolve(test,secondorder_y,mode='same')*convolve(test,[[-1,-1],[1,1]],mode='same')
        D[i-1][:,:,j-1]=temp*convolve(test,[[-1,-1],[1,1]],mode='same')*0.5+test


count=0
# locating the extrema and selecting them based on threshold
for i in range(1,int(extrema_volume+1)):
    # reconstructing x and y form the otave and the vloumeposition
    x = np.floor((extrema[4*(i-1)+2] - 1) / (n / (2 ** (extrema[4*(i-1)] - 2)))) +1
    y = np.remainder((extrema[4*(i-1)+2] - 1) ,(m / (2 ** (extrema[4*(i-1)] - 2)))) +1
    rx=int(x+1)
    ry=int(y+1)
    # getting the level 
    rz=extrema[4*(i-1 )+1]
    # print(rz)
    rz=int(rz)
    # print(rx)
    # print(ry)
    
    #getting the value of the extrema and comapring it with the threshold
    z=D[int(extrema[4*(i-1)])-1][rx-1,ry-1,rz]
    if(np.abs(z)<thershold):
        extrema[4*(i-1)+3]=0
        count+=1

# print(count)
# print(extrema.shape)
# Keeping the better extremas
idx=np.where(extrema == 0)
idx=idx[0]

idx=np.concatenate([idx,idx-1,idx-2,idx-3])
extrema=np.delete(extrema,idx)

# extreacitng the points sames as before and then plotting the new points
extrema_volume=len(extrema)/4
img_height, img_width = img.shape
extrema_values = extrema[2::4] # row col value
extrema_octaves = extrema[0::4] # octave value

x = np.floor((extrema_values - 1) / ((img_width) / (2 ** (extrema_octaves-2)))) +1
y = np.remainder((extrema_values - 1), ((img_height) / (2 ** (extrema_octaves-2))))+1

ry = y / (2 ** (octave - 1 - extrema_octaves))
rx = x / (2 ** (octave - 1 - extrema_octaves))


plt.figure(1)
plt.imshow(img,cmap='gray')
plt.scatter(ry,rx,marker='+',color='red')
plt.show()

count2=0

# reiterating through the filtered extremas and then find the double derivatives of the extremas and re thresholding the values?
for i in range(1,int(extrema_volume+1)):
    x = np.floor((extrema[4*(i-1)+2] - 1) / (n / (2 ** (extrema[4*(i-1)] - 2)))) +1
    y = np.remainder((extrema[4*(i-1)+2] - 1) ,(m / (2 ** (extrema[4*(i-1)] - 2)))) +1
    rx=int(x+1)
    ry=int(y+1)
    rz=extrema[4*(i-1 )+1]
    rz=int(rz)

    #calculating the doube derivatives by David Lowes Code
    Dxx=D[int(extrema[4*(i-1)])-1][rx-2,ry-1,rz]+D[int(extrema[4*(i-1)])-1][rx,ry-1,rz]-2*D[int(extrema[4*(i-1)])-1][rx-1,ry-1,rz]
    Dyy=D[int(extrema[4*(i-1)])-1][rx-1,ry-2,rz]+D[int(extrema[4*(i-1)])-1][rx-1,ry,rz]-2*D[int(extrema[4*(i-1)])-1][rx-1,ry-1,rz]
    Dxy=D[int(extrema[4*(i-1)])-1][rx-2,ry-2,rz]+D[int(extrema[4*(i-1)])-1][rx,ry,rz]*D[int(extrema[4*(i-1)])-1][rx-2,ry,rz]*D[int(extrema[4*(i-1)])-1][rx,ry-2,rz]
    deter=Dxx*Dyy-Dxy*Dxy
    R=(Dxx+Dyy)/deter
    R_threshold=(r+1)**2/r
    if(deter<0 or R_threshold<R):
        extrema[4*(i-1)+3]=0
        count2+=1
        
# print(extrema.shape)
# print(count2)
# extreacitng the points sames as before and then plotting the new points
idx=np.where(extrema == 0)
idx=idx[0]

idx=np.concatenate([idx,idx-1,idx-2,idx-3])
extrema=np.delete(extrema,idx)
# print(extrema.shape)
extrema_volume=len(extrema)/4
img_height, img_width = img.shape
extrema_values = extrema[2::4] # row col value
extrema_octaves = extrema[0::4] # octave value

x = np.floor((extrema_values - 1) / ((img_width) / (2 ** (extrema_octaves-2)))) +1
y = np.remainder((extrema_values - 1), ((img_height) / (2 ** (extrema_octaves-2))))+1

ry = y / (2 ** (octave - 1 - extrema_octaves))
rx = x / (2 ** (octave - 1 - extrema_octaves))


plt.figure(1)
plt.imshow(img,cmap='gray')
plt.scatter(ry,rx,marker='+',color='blue')
plt.show()