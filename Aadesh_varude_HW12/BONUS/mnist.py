import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import math
import datetime
import platform
import cv2
import numpy as np

#THIS MODEL WAS TRAINED ON GOOGLE COLAB
# mnist_dataset = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

# # Save image parameters to the constants that we will use later for data re-shaping and for model traning.
# (_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
# IMAGE_CHANNELS = 1

# print('IMAGE_WIDTH:', IMAGE_WIDTH)
# print('IMAGE_HEIGHT:', IMAGE_HEIGHT)
# print('IMAGE_CHANNELS:', IMAGE_CHANNELS)
# x_train_with_chanels = x_train.reshape(
#     x_train.shape[0],
#     IMAGE_WIDTH,
#     IMAGE_HEIGHT,
#     IMAGE_CHANNELS
# )

# x_test_with_chanels = x_test.reshape(
#     x_test.shape[0],
#     IMAGE_WIDTH,
#     IMAGE_HEIGHT,
#     IMAGE_CHANNELS
# )
# x_train_normalized = x_train_with_chanels / 255
# x_test_normalized = x_test_with_chanels / 255

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Convolution2D(
#     input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
#     kernel_size=5,
#     filters=8,
#     strides=1,
#     activation=tf.keras.activations.relu,
#     kernel_initializer=tf.keras.initializers.VarianceScaling()
# ))

# model.add(tf.keras.layers.MaxPooling2D(
#     pool_size=(2, 2),
#     strides=(2, 2)
# ))

# model.add(tf.keras.layers.Convolution2D(
#     kernel_size=5,
#     filters=16,
#     strides=1,
#     activation=tf.keras.activations.relu,
#     kernel_initializer=tf.keras.initializers.VarianceScaling()
# ))

# model.add(tf.keras.layers.MaxPooling2D(
#     pool_size=(2, 2),
#     strides=(2, 2)
# ))

# model.add(tf.keras.layers.Flatten())

# model.add(tf.keras.layers.Dense(
#     units=128,
#     activation=tf.keras.activations.relu
# ));

# model.add(tf.keras.layers.Dropout(0.2))

# model.add(tf.keras.layers.Dense(
#     units=10,
#     activation=tf.keras.activations.softmax,
#     kernel_initializer=tf.keras.initializers.VarianceScaling()
# ))
# tf.keras.utils.plot_model(
#     model,
#     show_shapes=True,
#     show_layer_names=True,
# )

# adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# model.compile(
#     optimizer=adam_optimizer,
#     loss=tf.keras.losses.sparse_categorical_crossentropy,
#     metrics=['accuracy']
# )
# log_dir=".logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# training_history = model.fit(
#     x_train_normalized,
#     y_train,
#     epochs=10,
#     validation_data=(x_test_normalized, y_test),
#     callbacks=[tensorboard_callback]
# )
# model_name = 'digits_recognition_cnn.h5'
# model.save(model_name, save_format='h5')



adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break


    img = cv2.resize(frame, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model = tf.keras.models.load_model('Aadesh_varude_HW12/BONUS/digits_recognition_cnn.h5', compile=False)  # Replace 'path_to_your_trained_model' with your model's path
    model.compile(
    optimizer=adam_optimizer,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'])


    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    x_test_normalized=img_array/255.0
    predictions_one_hot = model.predict([x_test_normalized])

    predictions = model.predict(img_array)
    predictions = np.argmax(predictions_one_hot, axis=1)
    
    cv2.putText(frame, str(predictions), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()