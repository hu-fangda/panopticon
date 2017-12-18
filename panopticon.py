"""
Code by: Fangda (Kenny) Hu

This program trains and tests on both a 10-category image classification task and a binary image classification task over the CIFAR-10 dataset of images.

Here is the summary of the performance of the Panopticon AI on both the 10-category image classification task and the binary image classification task:

Eval. results for the 10-category classifier:
10000/10000 [==============================] - 44s 4ms/step
The evaluation results are: [0.86565028581619263, 0.70810000000000005]!

Eval. results for the binary classifier:
10000/10000 [==============================] - 43s 4ms/step
The evaluation results are: [0.19041307733953, 0.93159999999999998]!


Considering the binary classifier takes less seconds per step, evaluates to a lower loss value, and displays
greater accuracy on the test data, the binary classification task is easier than classification into 10 categories

This is because output to the binary classification network uses a sigmoid activation function, while
output to the 10-categories network uses a softmax activation function. While both functions are functionally the same
(i.e. both classify the photos into one and only one category), the learning process requires significant amounts
of differentiation and mathmatically, exponential functions (e.g. sigmoid) are cheaper computationally to differentiate.

"""

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

#PART 1 - 10-category Classification Task. Start with importing CIFAR-10 data and converts to 1-hot format
def load_cifar10():
    #print("Starting data load...")
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    #computes ytrain_1hot
    ytrain_1hot = np.zeros((50000,10), dtype=np.int8)
    ytrain_2 = np.reshape(ytrain,50000)
    ytrain_index = np.arange(0,50000)

    ytrain_1hot[ytrain_index,ytrain_2] = 1

    #computes ytest_1hot
    ytest_1hot = np.zeros((10000, 10), dtype=np.int8)
    ytest_2 = np.reshape(ytest, 10000)
    ytest_index = np.arange(0, 10000)

    ytest_1hot[ytest_index, ytest_2] = 1

    #normalizes RBG data

    xtest = xtest/255.0
    xtrain = xtrain / 255.0

    return xtrain, ytrain_1hot, xtest, ytest_1hot

#PART 2 - Output reproduced in Panopticon_output_part2.txt
def build_multilayer_nn():
    nn = Sequential()
    nn.add(Flatten(input_shape=(32,32,3)))

    hidden = Dense(units=100, activation="relu", input_shape=(3072,))
    nn.add(hidden)

    output = Dense(units=10, activation="softmax")
    nn.add(output)

    return nn


def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)

#PART 3 - Output reproduced in Panopticon_output_part3.txt
"""
Changes made (versus Part 2):
Increased learning rate from 0.01 to 0.05
(while the higher learning rate means the model "abandons" old beliefs quicker could lead to overfitting, this compromise is needed
to accomodate the increase in training epochs and the limitations of my machine, which has no GPU)
Increased training epochs from 20 to 30
Reduced batch size from 32 to 16

Increased dropout rate on the 2nd dropout layer from 0.50 to 0.65 
(so as to reduce overfitting and make for a more robust model)

"""

def build_convolution_nn():
    nn = Sequential()
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32,32,3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    nn.add(Dropout(0.25))

    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(4, 4)))

    nn.add(Dropout(0.65))

    nn.add(Flatten(input_shape=(8, 8, 32)))
    nn.add(Dense(units=250, activation="relu", input_shape=(2048,)))
    nn.add(Dense(units=100, activation="relu", input_shape=(2048,)))

    nn.add(Dense(units=10, activation="softmax"))

    return nn
  #  pass


def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.05)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=16)

#Part 4 - Binary Classification Task with output reproduced in Panopticon_output_part4.txt. First, we must reimport the data and assign them one of two labels.
def get_binary_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # computes binary classifications. Animal == 1 and vehicle == 0
    for label1 in ytrain:
        if (label1[0] == 2 or label1[0] == 3 or label1[0] == 4 or label1[0] == 5 or label1[0] == 6 or label1[0] == 7):
            label1[0] = 1
        elif (label1[0] == 0 or label1[0] == 1 or label1[0] == 8 or label1[0] == 9):
            label1[0] = 0

    for label1 in ytest:
        if (label1[0] == 2 or label1[0] == 3 or label1[0] == 4 or label1[0] == 5 or label1[0] == 6 or label1[0] == 7):
            label1[0] = 1
        elif (label1[0] == 0 or label1[0] == 1 or label1[0] == 8 or label1[0] == 9):
            label1[0] = 0
    # normalizes RBG data
    xtest = xtest / 255.0
    xtrain = xtrain / 255.0
    print("Loaded binary cifar10 data!")
    return xtrain, ytrain, xtest, ytest

#Note: Part 4 uses the same optimized network structure and training parameters as Part 3

def build_binary_classifier():
    nn = Sequential()
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    nn.add(Dropout(0.25))

    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(4, 4)))

    nn.add(Dropout(0.65))

    nn.add(Flatten(input_shape=(8, 8, 32)))
    nn.add(Dense(units=250, activation="relu", input_shape=(2048,)))
    nn.add(Dense(units=100, activation="relu", input_shape=(2048,)))

    nn.add(Dense(units=1, activation="sigmoid"))

    return nn

def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.05)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=30, batch_size=16)

if __name__ == "__main__":
    #loading the NON-BINARY cifar10 data
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()

    #multilayer network testing code
    nn=build_multilayer_nn()
    nn.summary()
    train_multilayer_nn(nn, xtrain, ytrain_1hot)
    print("The evaluation results are: " + str(nn.evaluate(xtest, ytest_1hot)) + "!")


    #convolution network testing code

    nn2=build_convolution_nn()
    train_convolution_nn(nn2, xtrain, ytrain_1hot)
    print("The evaluation results are: " + str(nn2.evaluate(xtest, ytest_1hot)) + "!")

    #loading the BINARY cifar10 data
    #binary classification convolution network testing code
    xtrain1, ytrain1, xtest1, ytest1 = get_binary_cifar10()
    nbinary = build_binary_classifier()
    train_binary_classifier(nbinary,xtrain1,ytrain1)
    print("The evaluation results are: " + str(nbinary.evaluate(xtest1, ytest1)) + "!")

    # Write any additional code for testing and evaluation in this main section.