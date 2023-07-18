# ML algorithm to detect the number images

from tensorflow import keras
from tensorflow.keras import layers

# import a set of data
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# construction of the Neural Network
model = keras.Sequential([
	layers.Dense(512, activation="relu"),
	layers.Dense(10, activation="softmax")
])

model.compile(	optimizer="rmsprop",
				loss="sparse_categorical_crossentropy",
				metrics=["accuracy"])

#print(train_images.shape)

# transform in 2D array shape with the values \in [0,1]
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# training the model with the dataset
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# using the trained model to do make predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)

print(predictions[0].argmax(), test_labels[0])
