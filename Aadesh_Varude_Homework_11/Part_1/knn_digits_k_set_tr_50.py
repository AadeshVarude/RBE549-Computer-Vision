import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('Aadesh_Varude_Homework_11/digits.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array: its size will be (50,100,20,20)
x = np.array(cells)

# Now we prepare the training data and test data
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

k_list=np.arange(1,10,1)
accuracies=[]
for k in range(1,10,1):
  knn = cv.ml.KNearest_create()
  knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
  ret,result,neighbours,dist = knn.findNearest(test,k=k)
  matches = result==test_labels
  correct = np.count_nonzero(matches)
  accuracy = correct*100.0/result.size
  accuracies.append(accuracy)
  print( accuracy )


# plotting the points  
plt.plot(k_list, accuracies) 
  
# naming the x axis 
plt.xlabel('k - values') 
# naming the y axis 
plt.ylabel('accuracies - axis') 
  
# giving a title to my graph 
plt.title('Part1-2 graph for all values of k') 
  
# function to show the plot 
plt.show() 
