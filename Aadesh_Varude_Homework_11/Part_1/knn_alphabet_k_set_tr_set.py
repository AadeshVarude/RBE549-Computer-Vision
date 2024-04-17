import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data and convert the letters to numbers
data= np.loadtxt('Aadesh_Varude_Homework_11/letter+recognition/letter-recognition.data', dtype= 'float32', delimiter = ',',
                    converters= {0: lambda ch: ord(ch)-ord('A')})

x_list=np.arange(0.1,1,0.1)
results = {'split': [], 'k': [], 'accuracy': []}

# Now we prepare the training data and test data
for x_percent in x_list:
    # Split the dataset as per the percentage
    train = data[0:int(x_percent*data.shape[0]),:]
    test = data[int(x_percent*data.shape[0]):,:]

    # Split trainData and testData into features and responses
    responses, trainData = np.hsplit(train,[1])
    labels, testData = np.hsplit(test,[1])
    k_list=np.arange(1,10,1)
    for k in range(1,10,1):
        knn = cv.ml.KNearest_create()
        knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
        ret, result, neighbours, dist = knn.findNearest(testData, k=k)
        correct = np.count_nonzero(result == labels)
        accuracy = correct*100.0/result.size
        results['split'].append(x_percent)
        results['k'].append(k)
        results['accuracy'].append(accuracy)
        print("accuracy :",accuracy)



df = pd.DataFrame(results)
for split in x_list:
    data = df[df['split'] == split]
    plt.plot(data['k'].values, data['accuracy'].values, label=f'Train size: {int(split)}%')


plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different Train/Test Splits and k Values')
plt.legend()
plt.show()