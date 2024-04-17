import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing the images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Must define the input shape in the first layer of the neural network
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPooling2D(pool_size=3),
                                    tf.keras.layers.Flatten(input_shape=(28, 28)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10) 

loss,accuracy=model.evaluate(test_images, test_labels) # Evaluating the trained model it rovide accuracy of 86.9 percent
print("Loss on the testing data :",loss,"Accuracy on the testing data :",accuracy*100)