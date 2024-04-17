import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread('Aadesh_Varude_Homework_11/part3_img.png')

# Reshape the image
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# Define criteria and values of K
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0) # Default criteria
criteria_1=(cv.TERM_CRITERIA_MAX_ITER, 50, 1.0) # My new criteria
k_values = [2, 3, 5, 10, 20, 40] # Values assigned by professor
k_values_1 = [3, 5, 10, 30, 50, 70] # Self K values

# Set up the subplot grid
rows = 2
cols = 3
fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

for i, K in enumerate(k_values):
    # Apply kmeans
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Convert back into uint8 and reshape the image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)
    
    # Plot the images
    ax = axs[i // cols, i % cols]
    ax.imshow(cv.cvtColor(res2, cv.COLOR_BGR2RGB))
    ax.set_title(f'K = {K}')
    ax.axis('off')

plt.tight_layout()
plt.show()