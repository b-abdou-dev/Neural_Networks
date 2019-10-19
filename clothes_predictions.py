
import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


## import fashion_mnist datasets that is available in keras library

data = keras.datasets.fashion_mnist

# print(data)

## split data into training set and testing set
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot']

## normlizing value of image
train_images = train_images/255
test_images = test_images/255


## setting model parameters
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

## training the model
model.fit(train_images, train_labels, epochs=10)

# ## getting the accuarcy of the model
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("Tested Acc:", test_acc)

prediction = model.predict(test_images)

# image 1 prediction value
print("predictions of the first image is:", prediction[0])


# all images predictions
for i in range(len(prediction)):
    print("the image", i, " is predicted to be ", class_names[np.argmax(prediction[i])])

for i in range(len(prediction)):
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Predicted: " + class_names[np.argmax(prediction[i])])
    plt.show()