import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd



img = cv.imread('Aadesh_Varude_Homework_11/digits.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array: its size will be (50,100,20,20)
x = np.array(cells)

# All possible values of k and train/testing datasets
# print(x.shape)
x_list=np.arange(10,100,10)
results = {'split': [], 'k': [], 'accuracy': []}
# # Now we prepare the training data and test data
for x_percent in x_list:
  train = x[:,:x_percent].reshape(-1,400).astype(np.float32) # Size = (2500,400)
  test = x[:,x_percent:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
  a = np.arange(10)
  train_labels = np.repeat(a,train.shape[0]/10)[:,np.newaxis]
  test_labels = np.repeat(a,test.shape[0]/10)[:,np.newaxis]
  # print(train.shape)
  # print(train_labels.shape)
  k_list=np.arange(1,10,1)
  for k in range(1,10,1):
    knn = cv.ml.KNearest_create()
    knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
    ret,result,neighbours,dist = knn.findNearest(test,k=k)
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    results['split'].append(x_percent)
    results['k'].append(k)
    results['accuracy'].append(accuracy)
    print( accuracy )


df = pd.DataFrame(results)
for split in x_list:
    data = df[df['split'] == split]
    plt.plot(data['k'].values, data['accuracy'].values, label=f'Train size: {int(split)}%')

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different Train/Test Splits and k Values')
plt.legend()
plt.show()