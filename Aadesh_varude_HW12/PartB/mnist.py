import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
#Loading and normalizing the data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Defining the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
#Defining the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizers = {
    'SGD': SGD(),
    'Adam': Adam(),
    'RMSprop': RMSprop()
}

results = {}

for optimizer_name, optimizer in optimizers.items():

  model.compile(optimizer=optimizer_name,
              loss=loss_fn,
              metrics=['accuracy'])

  print(f'Training with optimizer: {optimizer_name}')
  history = model.fit(x_train, y_train, epochs=5)
  results[optimizer_name] = history

# Plot training accuracy and loss graphs
plt.figure(figsize=(12, 6))

for optimizer_name, history in results.items():
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label=f'{optimizer_name} Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label=f'{optimizer_name} Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()

# Report results
for optimizer_name, history in results.items():
    print(f'\nOptimizer: {optimizer_name}')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')