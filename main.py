import numpy as np
from matplotlib import pyplot as plt
from random import randrange
from helper import *
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2


# SET PARAMETERS OF THE MODEL
NUM_EPOCHS = 10
LR_INIT = 1e-2
BATCH_SIZE = 32


# LOADING THE DATA
((X_train, Y_train), (X_test, Y_test)) = fashion_mnist.load_data()

plt.figure()
rd = X_train.shape[0]-1
plt.title('Sample %s' %(rd))
plt.imshow(X_train[randrange(rd)])


# RESHAPING DATA

# converting the image to correct format. HEre only one channel (no RGB)
if K.image_data_format() == "channels_first":
    X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

else:
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))


# NORMALIZING DATA

# scaling the data to the range [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# one-hot encode the training and testing labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# initializing the label names
labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]


# INITIALIZING THE MODEL AND THE OPTIMIZER,

print("Compiling the model")
opt = SGD(lr=LR_INIT , momentum=0.9, decay=LR_INIT / NUM_EPOCHS)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# TRAINING THE MODEL

print("Training the model")
History = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)


# PREDICTING THE MODEL ON THE TEST DATASET

preds = model.predict(X_test)
print("Model performence:")
print(classification_report(Y_test.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

# ploting the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss and Accuracy on Dataset")
plt.plot(np.arange(0, N), History.history["loss"], label="training loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="testing loss")
plt.plot(np.arange(0, N), History.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0, N), History.history["val_accuracy"], label="testing accuracy")
plt.xlabel("Number of epoches")
plt.ylabel("Loss/Accuracy")
plt.legend()


# VISUALIZING THE PREDICTIONS

images = []
for i in np.random.choice(np.arange(0, len(Y_test)), size=(16,)): # selecting 16 random clothe to classify
    # classifying the intput data
    probs = model.predict(X_test[np.newaxis, i])
    prediction = probs.argmax(axis=1)
    label = labelNames[prediction[0]]

    if K.image_data_format() == "channels_first":
        image = (X_test[i][0] * 255).astype("uint8")

    else:
        image = (X_test[i] * 255).astype("uint8")

    color = (0, 255, 0) # green is correct
    if prediction[0] != np.argmax(Y_test[i]): # if the predction is wrong
        color = (0, 0, 255)

    image = cv2.merge([image] * 3) # merge channels into one image
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR) # resize the image from 28x28 to 96x96
    cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    images.append(image)

# finalizing the montage
montage = build_montages(images, (96, 96), (4, 4))[0]
cv2.imshow("Fashion MNIST", montage)
cv2.waitKey(0)


