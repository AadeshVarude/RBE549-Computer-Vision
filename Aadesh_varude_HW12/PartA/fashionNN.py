import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing the images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# DeFining the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Setting the optimizer as Adam loss as the cross entropy loss and the metrics for measurment as accuracy
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5) # Training the model for 5 epoches
loss,accuracy=model.evaluate(test_images, test_labels) # Evaluating the trained model it rovide accuracy of 86.9 percent
print("Loss on the testing data :",loss,"Accuracy on the testing data :",accuracy*100)
