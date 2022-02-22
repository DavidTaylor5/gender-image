"""
David Taylor 1/25/21
I am working on determining a person's gender based on their Facebook profile pictures.
I plan on using a simple CNN. -> https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
If I have time I will use the CNN from  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
from tabnanny import verbose
from unicodedata import name
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Dense, Dropout, LayerNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

#import cv2 

import matplotlib.pyplot as plt
import gzip
import sys
import pickle as cPickle

from sklearn import metrics

import os
from os import listdir

import pandas as pd
import numpy as np
import math
# I will have to change path/getting command line argument for program.
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import load_img
import warnings
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


"""For this current iteration I plan on using down scaled (reshaped?) from (272, 200) RGB to (32, 32) GrayScale 
to be in line with the handwriting CNN. The features Numpy Array should be in form (9500, 32, 32, 1) and the
labels Numpy array should be in the form (9500, 2). I have found that full sized images w/ RGB to creat a 
NumpyArray that would be 11GiB! Way too big. I'm not sure color will have an affect on gender. """


#okay I think this data stuff might be ready for simple CNN model. 


# Creating the model
#tf.keras.layers.Conv2D(filterns, kernel_size, strides=(1, 1), padding="valid", data_format=None, dilation_rate=(1, 1))
#How should I know what to make the filter size and kernel_size in CNN? Is this just a hyperparameter?
def valueGender(value):
	if(value == 0):
		return "male"
	elif(value == 1):
		return "female"
	else:
		return "error in gender classification"

def setGenders(array):
	genders = []
	for i in range(0, len(array)):
		genders.append(valueGender(array[i]))	


def gen_cnn(num_classes): #num_classes = 2 male and female
	#Create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(128, 128, 1), activation='relu')) #784 PIXELS 28 X 28 | 30 filters (5, 5) kernel size eqauals 750, maybe 1632 (100, 100)? #size account
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def gen_cnn_second(num_classes): #num_classes = 2 male and female
	model = Sequential()
	model.add(Conv2D(input_shape=(128, 128, 1), filters=32, kernel_size=3, strides=4, padding='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model.add(LayerNormalization())
	model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	model.add(LayerNormalization())
	model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(LayerNormalization())

	model.add(Flatten())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(rate=0.25))
	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(rate=0.25))
	model.add(Dense(units=2, activation='softmax'))

	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


if __name__== "__main__":
	mydirectory = 'imgProcessedGood/' #possible place for sys.arg[0]

	profileData = pd.read_csv('training/profile/profile.csv')
	###
	#features = np.empty((9500, 128, 128, 1)) #size account

	images = []
	imgId = []
	genders = []

	for image in os.listdir(mydirectory):
		img = load_img( mydirectory + image, color_mode="grayscale", target_size=(128, 128))
		img_numpy_array = img_to_array(img)

		images.append(img_numpy_array)

		myId = image[:-4] #removes .jpg
		imgId.append(myId)
		#print(myId)

	#now I need to make arrays into numpy array (2825, 128, 128, 1)
	features = np.array(images)
	print(features.shape) #correct shape (2825, 128, 128, 1)
	#wow

	for id in imgId:
		currRow = profileData[profileData['userid'] == id]
		# print(id)#looks like I got the correct row
		#print(currRow) #looks correct
		index = profileData[profileData['userid'] == id].index[0]
		#print(index) #should be 2457
		currGender = profileData.loc[index, 'gender'] #okay I have gotten the first label/correct
		genders.append(currGender)
		#print(currGender)


	#normalize by dividing by 255, range of grayscale #this might not be the correct type of grayscale
	features = features / 255

	# put labels into category vector
	genders = np.array(genders)
	genders = np_utils.to_categorical(genders)

	#features should be 9500 different images -> test 2850 (30%) train 6650 (70%)
	# split features and labels into train and test
	split = math.floor(.75 * len(features))


	X_train = features[0:split, :]  
	X_test  = features[split:, :]
	y_train = genders[0:split, :]
	y_test = genders[split:, :]

	print("Done processing datasets!")

	print("Working on CNN...")
	#build the model
	model = gen_cnn(2)
	# fit the model
	model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
	# final evaluation of the model



	predictions = model.predict(X_test) #predictions based on input. Need to get ids from input...
	predictions = np.argmax(predictions, axis=1) # 0 -> male, 1-> female
	#	#predictions = math.ceil(predictions)
	print(predictions.shape)
	print(type(predictions))

	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Overall Accuracy: ", scores)
	print("GrayScale 128 X 128 CNN Error: %.2f%%" % (100-scores[1]*100))


#predictions = model.predict(X_test)
"""
Hopefully these scores are accurate?
 1/34 [..............................] - ETA: 16s - loss: 0.6934 - accuracy: 0.51502022-01-25 20:06:22.503110: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 18816000 exceeds 10% of free 
system memory.
34/34 [==============================] - 3s 64ms/step - loss: 0.6781 - accuracy: 0.5758 - val_loss: 0.6804 - val_accuracy: 0.5705
Epoch 2/10
34/34 [==============================] - 2s 58ms/step - loss: 0.6644 - accuracy: 0.5916 - val_loss: 0.6614 - val_accuracy: 0.5765
Epoch 3/10
34/34 [==============================] - 2s 61ms/step - loss: 0.6497 - accuracy: 0.6062 - val_loss: 0.6468 - val_accuracy: 0.6007
Epoch 4/10
34/34 [==============================] - 2s 63ms/step - loss: 0.6279 - accuracy: 0.6337 - val_loss: 0.6933 - val_accuracy: 0.5912
Epoch 5/10
34/34 [==============================] - 2s 64ms/step - loss: 0.6178 - accuracy: 0.6405 - val_loss: 0.6155 - val_accuracy: 0.6453
Epoch 6/10
34/34 [==============================] - 2s 61ms/step - loss: 0.5950 - accuracy: 0.6704 - val_loss: 0.6865 - val_accuracy: 0.6147
Epoch 7/10
34/34 [==============================] - 2s 65ms/step - loss: 0.6117 - accuracy: 0.6550 - val_loss: 0.6138 - val_accuracy: 0.6446
Epoch 8/10
34/34 [==============================] - 2s 70ms/step - loss: 0.5832 - accuracy: 0.6809 - val_loss: 0.6066 - val_accuracy: 0.6533
Epoch 9/10
34/34 [==============================] - 2s 67ms/step - loss: 0.5655 - accuracy: 0.6988 - val_loss: 0.6137 - val_accuracy: 0.6530
Epoch 10/10
34/34 [==============================] - 2s 66ms/step - loss: 0.5690 - accuracy: 0.6980 - val_loss: 0.6293 - val_accuracy: 0.6361
Overall Accuracy:  [0.6292510628700256, 0.6361403465270996]
GrayScale 32 X 32 CNN Error: 36.39%

63% accuracy!

second run
Epoch 10/10
34/34 [==============================] - 2s 64ms/step - loss: 0.5743 - accuracy: 0.6904 - val_loss: 0.6246 - val_accuracy: 0.6456
(2850, 2)
<class 'numpy.ndarray'>
Overall Accuracy:  [0.6246270537376404, 0.6456140279769897] -> I guess this is my best score
GrayScale 32 X 32 CNN Error: 35.44%
"""

"""
#using larger size images -> reshape to be 128, 128) I'm now working with pictures that have 16 times as many pixels, what is outcome?
 1/34 [..............................] - ETA: 50s - loss: 0.6962 - accuracy: 0.47502022-02-03 14:25:05.566401: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 369024000 exceeds 10% of free system memory.  
2022-02-03 14:25:05.993333: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 369024000 exceeds 10% of free system memory.
 2/34 [>.............................] - ETA: 26s - loss: 0.7593 - accuracy: 0.49752022-02-03 14:25:06.386484: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 369024000 exceeds 10% of free system memory.  
34/34 [==============================] - 30s 872ms/step - loss: 0.6849 - accuracy: 0.5698 - val_loss: 0.6754 - val_accuracy: 0.5761
Epoch 2/10
34/34 [==============================] - 29s 854ms/step - loss: 0.6670 - accuracy: 0.5842 - val_loss: 0.6712 - val_accuracy: 0.5779
Epoch 3/10
34/34 [==============================] - 29s 854ms/step - loss: 0.6539 - accuracy: 0.6044 - val_loss: 0.6600 - val_accuracy: 0.5916
Epoch 4/10
34/34 [==============================] - 29s 852ms/step - loss: 0.6233 - accuracy: 0.6474 - val_loss: 0.6772 - val_accuracy: 0.6021
Epoch 5/10
34/34 [==============================] - 29s 854ms/step - loss: 0.5921 - accuracy: 0.6777 - val_loss: 0.6471 - val_accuracy: 0.6316
Epoch 6/10
34/34 [==============================] - 29s 850ms/step - loss: 0.5387 - accuracy: 0.7278 - val_loss: 0.6566 - val_accuracy: 0.6298
Epoch 7/10
34/34 [==============================] - 29s 850ms/step - loss: 0.4894 - accuracy: 0.7620 - val_loss: 0.6755 - val_accuracy: 0.6411
Epoch 8/10
34/34 [==============================] - 29s 856ms/step - loss: 0.4288 - accuracy: 0.8029 - val_loss: 0.7049 - val_accuracy: 0.6147
Epoch 9/10
34/34 [==============================] - 29s 853ms/step - loss: 0.3558 - accuracy: 0.8525 - val_loss: 0.7815 - val_accuracy: 0.6221
Epoch 10/10
34/34 [==============================] - 29s 865ms/step - loss: 0.2837 - accuracy: 0.8854 - val_loss: 0.8995 - val_accuracy: 0.6277
(2850,)
<class 'numpy.ndarray'>
Overall Accuracy:  [0.8995048403739929, 0.6277192831039429]
GrayScale 128 X 128 CNN Error: 37.23%
# Whoa almost 90% lets go baby

#size 192 largest possible? test test
 1/34 [..............................] - ETA: 3:41 - loss: 0.6985 - accuracy: 0.48502022-02-03 14:34:47.175509: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 848256000 exceeds 10% of free system memory.
2022-02-03 14:34:48.220109: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 424128000 exceeds 10% of free system memory.
34/34 [==============================] - 77s 2s/step - loss: 0.7245 - accuracy: 0.5620 - val_loss: 0.6775 - val_accuracy: 0.5761
Epoch 2/10
34/34 [==============================] - 70s 2s/step - loss: 0.6663 - accuracy: 0.6017 - val_loss: 0.6768 - val_accuracy: 0.5902
Epoch 3/10
34/34 [==============================] - 70s 2s/step - loss: 0.6413 - accuracy: 0.6286 - val_loss: 0.6599 - val_accuracy: 0.6000
Epoch 4/10
34/34 [==============================] - 70s 2s/step - loss: 0.5953 - accuracy: 0.6830 - val_loss: 0.6746 - val_accuracy: 0.6095
Epoch 5/10
34/34 [==============================] - 70s 2s/step - loss: 0.5409 - accuracy: 0.7247 - val_loss: 0.6884 - val_accuracy: 0.6056
Epoch 6/10
34/34 [==============================] - 70s 2s/step - loss: 0.4807 - accuracy: 0.7705 - val_loss: 0.7342 - val_accuracy: 0.6035
Epoch 7/10
34/34 [==============================] - 70s 2s/step - loss: 0.4096 - accuracy: 0.8119 - val_loss: 0.7909 - val_accuracy: 0.6039
Epoch 8/10
34/34 [==============================] - 70s 2s/step - loss: 0.3351 - accuracy: 0.8489 - val_loss: 0.9157 - val_accuracy: 0.6116
Epoch 9/10
34/34 [==============================] - 70s 2s/step - loss: 0.2536 - accuracy: 0.8940 - val_loss: 1.0103 - val_accuracy: 0.6014
Epoch 10/10
34/34 [==============================] - 70s 2s/step - loss: 0.1960 - accuracy: 0.9232 - val_loss: 1.1389 - val_accuracy: 0.6056
(2850,)
<class 'numpy.ndarray'>
Overall Accuracy:  [1.1389193534851074, 0.6056140065193176]
GrayScale 128 X 128 CNN Error: 39.44%


Now that I have my accuracy value I might need to keep track of ids so that I can output? Should still be in order?
I need to make sure that I train on my training set and I set values for my non training set based on some path.... 
"""

"""
I need to be able to make the output
"""

"""
Using preprocessed training data
Epoch 1/10
67/67 [==============================] - 11s 154ms/step - loss: 0.6325 - accuracy: 0.6714 - val_loss: 0.5393 - val_accuracy: 0.7327
Epoch 2/10
67/67 [==============================] - 10s 147ms/step - loss: 0.5532 - accuracy: 0.7361 - val_loss: 0.5139 - val_accuracy: 0.7525
Epoch 3/10
67/67 [==============================] - 10s 147ms/step - loss: 0.5146 - accuracy: 0.7521 - val_loss: 0.5276 - val_accuracy: 0.7397
Epoch 4/10
67/67 [==============================] - 10s 148ms/step - loss: 0.4826 - accuracy: 0.7762 - val_loss: 0.4852 - val_accuracy: 0.7737
Epoch 5/10
67/67 [==============================] - 10s 148ms/step - loss: 0.4309 - accuracy: 0.8107 - val_loss: 0.5342 - val_accuracy: 0.7369
Epoch 6/10
67/67 [==============================] - 10s 147ms/step - loss: 0.4164 - accuracy: 0.8239 - val_loss: 0.4979 - val_accuracy: 0.7737
Epoch 7/10
67/67 [==============================] - 10s 147ms/step - loss: 0.3618 - accuracy: 0.8343 - val_loss: 0.5352 - val_accuracy: 0.7751
Epoch 8/10
67/67 [==============================] - 10s 148ms/step - loss: 0.3103 - accuracy: 0.8725 - val_loss: 0.5359 - val_accuracy: 0.7737
Epoch 9/10
67/67 [==============================] - 10s 149ms/step - loss: 0.2765 - accuracy: 0.8810 - val_loss: 0.7420 - val_accuracy: 0.6987
Epoch 10/10
67/67 [==============================] - 10s 147ms/step - loss: 0.2318 - accuracy: 0.9013 - val_loss: 0.6708 - val_accuracy: 0.7553
(707,)
<class 'numpy.ndarray'>
Overall Accuracy:  [0.6707913279533386, 0.7553040981292725]
GrayScale 128 X 128 CNN Error: 24.47%

Accuracy of 75% Best so far. (Gender)
I could test out different facial classifiers. 
I could even loop through classifiers until I find one that can identity a face to get more training examples
I could generate more images from by current 'good' training set.
I could work on normalizing for the CNN

REMOVED THE SQUARES ON EYES FOR IMGPROCESSEDGOOD
Epoch 1/10
67/67 [==============================] - 105s 2s/step - loss: 0.6036 - accuracy: 0.6868 - val_loss: 0.5927 - val_accuracy: 0.7115
Epoch 2/10
67/67 [==============================] - 99s 1s/step - loss: 0.5549 - accuracy: 0.7245 - val_loss: 0.7021 - val_accuracy: 0.5686
Epoch 3/10
67/67 [==============================] - 97s 1s/step - loss: 0.5017 - accuracy: 0.7608 - val_loss: 0.4778 - val_accuracy: 0.7808
Epoch 4/10
67/67 [==============================] - 100s 1s/step - loss: 0.4564 - accuracy: 0.7972 - val_loss: 0.4798 - val_accuracy: 0.7935
Epoch 5/10
67/67 [==============================] - 97s 1s/step - loss: 0.3936 - accuracy: 0.8274 - val_loss: 0.4603 - val_accuracy: 0.7949
Epoch 6/10
67/67 [==============================] - 98s 1s/step - loss: 0.3484 - accuracy: 0.8528 - val_loss: 0.4859 - val_accuracy: 0.7935
Epoch 7/10
67/67 [==============================] - 99s 1s/step - loss: 0.3012 - accuracy: 0.8698 - val_loss: 0.4963 - val_accuracy: 0.7949
Epoch 8/10
67/67 [==============================] - 98s 1s/step - loss: 0.2649 - accuracy: 0.8892 - val_loss: 0.5052 - val_accuracy: 0.8020
Epoch 9/10
67/67 [==============================] - 98s 1s/step - loss: 0.1921 - accuracy: 0.9231 - val_loss: 0.6085 - val_accuracy: 0.7737
Epoch 10/10
67/67 [==============================] - 97s 1s/step - loss: 0.1541 - accuracy: 0.9392 - val_loss: 0.6006 - val_accuracy: 0.7977
(707,)
<class 'numpy.ndarray'>
Overall Accuracy:  [0.6005898714065552, 0.7977369427680969]
GrayScale 128 X 128 CNN Error: 20.23% #this will be worse for pictures that are not good training images. 
#79% accuracy with only good images. 

#training with good faces and unsure faces. 

"""