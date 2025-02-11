The dataset used for this project was downloaded from Kaggle.

## DESCRIPTION FROM KAGGLE

BIRDS 525 SPECIES- IMAGE CLASSIFICATION
525 species, 84635 train, 2625 test, 2625 validation images 224X224X3 jpg

https://www.kaggle.com/datasets/gpiosenka/100-bird-species/data

Data set of 525 bird species. 84635 training images, 2625 test images(5 images per species) and 2625 validation images(5 images per species. This is a very high quality dataset where there is only one bird in each image and the bird typically takes up at least 50% of the pixels in the image. As a result even a moderately complex model will achieve training and test accuracies in the mid 90% range.

Note: all images are original and not created by augmentation
All images are 224 X 224 X 3 color images in jpg format. Data set includes a train set, test set and validation set. Each set contains 525 sub directories, one for each bird species.

The data structure is convenient if you use the Keras ImageDataGenerator.flow_from_directory to create the train, test and valid data generators. The data set also include a file birds.csv. This cvs file contains 5 columns. The filepaths column contains the relative file path to an image file. The labels column contains the bird species class name associated with the image file. The scientific label column contains the latin scientific name for the image. The data set column denotes which dataset (train, test or valid) the filepath resides in. The class_id column contains the class index value associated with the image file's class.

## CNN

In the cnn.py file there are 2 cnn. The BCNN (Bird CNN) is a simple convolution neural network with Batch Normalization and Global Pooling. The BCNN_Red is a more advanced convolution neural network that also contains SE (Squeeze and Excitation) Blocks. It has many more layers and uses LeakyRELU as an activation function.
