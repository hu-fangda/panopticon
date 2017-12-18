# Panopticon
A Python 3 image classifier (trains and tests on CIFAR-10 data) utilizing the Keras Neural Network API with Tensorflow backend

Dependencies: Numpy, TensorFlow and Keras must be installed. For this program, the CPU version of Tensorflow is good enough. 

Tensorflow: https://www.tensorflow.org/install/
Keras: https://keras.io/#installation 

This program takes command line arguments as follows:

python panopticon.py 

----------
Inputs:
Training, development and testing of the Panopticon's neural network is done on the the CIFAR-10 image data set described here: https://www.cs.toronto.edu/~kriz/cifar.html. We use supervised learning; the CIFAR-10 data is already pre-labelled. 

The dataset as a whole contains 60,000 images represented as one 32x32x3 cube each. We split the whole dataset in a 50,000-image training set and a 10,000-image testing set. Training and development is never done on the testing set to avoid overfitting. 

## 10-category Classification Task
Each image falls into one of ten mutually exclusive categories:

Numeric ID | Category Name

0	| airplane

1	| automobile

2	| bird

3	| cat

4	| deer

5	| dog

6	| frog

7	| horse

8	| ship

9	| truck

## Binary Classification Task
Each image falls into one of two mutually exclusive categories:

Numeric ID | Category Name

0	| animal

1	| vehicle

----------
Outputs:
The Panopticon will build the relevant neural networks and start training them on the training dataset. As the training happens, the status will be updated and printed to console. After training is complete, the data is evaluated on the test dataset and the accuracy of the Panopticon will be printed to console. 

Outputs produced by a sample run of Parts 2, 3 and 4 can be found in Panopticon_output_part2.txt, Panopticon_output_part3.txt, and Panopticon_output_part4.txt - which are included in this repository. 

----------
Implementation details:

**Part 1** simply loads the CIFAR-10 dataset into our model, and populates the training and test data vectors. xtrain, ytrain, xtest, and ytest are four numpy n-dimension arrays. For example, xtrain[0] is an image of a frog and therefore ytrain[0] contains the value 6. The function returns 4 numpy arrays: xtrain, ytrain_1hot, xtest, ytest_1hot. ytrain_1hot and ytest_1hot are the image labels in 1-hot representation format (i.e. if the class label is 2, the 1-hot vector is [0 0 1 0 0 0 0 0 0 0]).

**Part 2** demonstrates a multilayer neural network with a single hidden layer, specifically a dense layer with 100 neurons. The rectified linear unit function is the chosen activation function for this layer. The output layer is a dense layer containing 10 neurons. The Softmax function is the chosen activation function for this layer. Pre-processing would have flattened the 32x32x3 array into a size-3072 vector. Run the method summary() to output a printed summary of the network topology. 

**Part 3** demonstrates a Convolution Neural Network (CNN) of the following topology:

Two convolution layers - each with 32 feature maps with a filter size of 3x3 and rectified linear unit activation. 

One pooling layer reducing each feature map by a factor of (2x2)

One drop-out layer that drops 25% of the units at random. 

Two convolution layers - each with 32 feature maps with a filter size of 3x3 and rectified linear unit activation. 

Another pooling layer, that reduces the size of each feature map by a factor of (4x4)

One drop-out layer that drops 65% of the units. 

The above outputs into a regular multilayer neural network. This network has two hidden layers of size 250 and 100, both with rectified linear unit activation. The output layer is a dense layer of size 10.

**Part 4** demonstrates the CNN (with the same topology) from Part 3, except on a binary classification task. 
