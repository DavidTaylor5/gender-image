"""
David Taylor 1/25/21
I am working on determining a person's age/gender based on their Facebook profile pictures.
I plan on using a simple CNN. -> https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
If I have time I will use the CNN from  https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

"""
import profile
from tabnanny import verbose
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import matplotlib.pyplot as plt
import gzip
import sys
import pickle as cPickle

from sklearn import metrics
import os
import pandas as pd
import numpy as np
import math
# I will have to change path/getting command line argument for program.
from keras.preprocessing.image import load_img
import warnings
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from xml.dom import minidom

import imgGender
import imgAge

def getAverages(profileData): #0.0 -> male, 1.0-> female
	opeAvg = profileData['ope'].mean()
	conAvg = profileData['con'].mean()
	extAvg = profileData['ext'].mean()
	agrAvg = profileData['agr'].mean()
	neuAvg = profileData['neu'].mean()
	genderAvg = profileData['gender'].mean()
	if genderAvg >= 0.5:
		gender = "female"
	else:
		gender = "male"
	twentyFour, thirtyFour, fourtyNine, fiftyPlus = 0, 0, 0, 0
	for x in range (0, profileData.shape[0]):
		age = profileData.loc[x]['age']
		if(age < 25):
			twentyFour += 1
		elif(age >= 25 and age < 34):
			thirtyFour += 1
		elif(age >=35 and age < 50):
			fourtyNine += 1
		else:
			fiftyPlus += 1
	ageRange = "none"
	maxRange = max(twentyFour, thirtyFour, fourtyNine, fiftyPlus)
	if (maxRange == twentyFour):
		ageRange = "xx-24"
	elif (maxRange == thirtyFour):
		ageRange = "25-34"
	elif (maxRange == fourtyNine):
		ageRange = "35-49"
	elif (maxRange == fiftyPlus):
		ageRange = "50-xx"
	else:
		print("Error in finding average age range") 
	
	print("This is the gender: ", gender)
	print("This is the age range: ", ageRange)
	print("This is the opeAvg: ", opeAvg)
	print("This is the conAvg: ", conAvg)
	print("This is the extAvg: ", extAvg)
	print("This is the agrAvg: ", agrAvg)
	print("This is the neuAvg: ", neuAvg)
	
	return gender, ageRange, opeAvg, conAvg, extAvg, agrAvg, neuAvg

def createFile(identify, ageGroup, gender, extrovert, neurotic, agreeable, conscientious, openness, outputPath):
	root = minidom.Document()
	xml = root.createElement('user')
	root.appendChild(xml)
	
	xml.setAttribute('id', identify)
	xml.setAttribute('age_group', ageGroup)
	xml.setAttribute('gender', gender)
	xml.setAttribute('extrovert', extrovert)
	xml.setAttribute('neurotic', neurotic)
	xml.setAttribute('agreeable', agreeable)
	xml.setAttribute('conscientious', conscientious)
	xml.setAttribute('open', openness)

	#xml_str = root.toprettyxml(indent ="\t")
	xml_str = root.toprettyxml(indent ="\t", newl="\n", encoding = None)

	save_path_file = outputPath + identify + ".xml"
	#save_path_file = "tester.xml"
	
	with open(save_path_file, "w") as f:
		f.write(xml_str)

if __name__ == "__main__":
    inputArg = sys.argv[2] #location of public test data ... I want to use the profile.csv to make predictions
    outputArg = sys.argv[4] #location of output/xml

    trainDirectory = 'training/image/'
    trainProfile = pd.read_csv("training/profile/profile.csv")

    #######################################     determine gender first    ######################################
    featuresTrain = np.empty((trainProfile.shape[0], 128, 128, 1))
    labelsTrain = []

    for index in range(0, trainProfile.shape[0]):
        userid = trainProfile.loc[index, 'userid']

        img = load_img(trainDirectory + userid + '.jpg', color_mode="grayscale", target_size=(128, 128)) #size account

        img_numpy_array = img_to_array(img)

        featuresTrain[index] = img_numpy_array

        gender = int(trainProfile.loc[index, 'gender'])

        labelsTrain.append(gender)
    
    featuresTrain = featuresTrain / 255

    # put labels into category vector
    labelsTrain = np.array(labelsTrain)
    labelsTrain = np_utils.to_categorical(labelsTrain)

    #features should be 9500 different images -> test 2850 (30%) train 6650 (70%)
	# split features and labels into train and test
    X_train = featuresTrain[0:6650, :]  
    X_test  = featuresTrain[6650:, :]
    y_train = labelsTrain[0:6650, :]
    y_test = labelsTrain[6650:, :]

    print("Done processing training dataset!")

    testProfile = pd.read_csv(inputArg + "profile/profile.csv")
    featuresTest = np.empty((testProfile.shape[0], 128, 128, 1))

    for index in range(0, testProfile.shape[0]):
        userid = testProfile.loc[index, 'userid']

        img = load_img(inputArg + "image/" + userid + '.jpg', color_mode="grayscale", target_size=(128, 128)) #size account

        img_numpy_array = img_to_array(img)

        featuresTest[index] = img_numpy_array

    featuresTest = featuresTest / 255

    print("Done processing testing dataset!")

    ageModel = imgGender.gen_cnn(2)

    ageModel.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=200)

    predictions = ageModel.predict(featuresTest)
    predictions = np.argmax(predictions, axis=1)

    scores = ageModel.evaluate(X_test, y_test, verbose=0)
    print("GrayScale 128 X 128 CNN Error: %.2f%%" % (100-scores[1]*100))

    gender_results = imgGender.setGenders(predictions) 
    #####################################################################################################
    ########################## Age Group Predictions ####################################################
    featuresTrain = np.empty((trainProfile.shape[0], 128, 128, 1))
    labelsTrain = []

    for index in range(0, trainProfile.shape[0]):
        userid = trainProfile.loc[index, 'userid']

        img = load_img(trainDirectory + userid + '.jpg', color_mode="grayscale", target_size=(128, 128)) #size account

        img_numpy_array = img_to_array(img)

        featuresTrain[index] = img_numpy_array

        age = int(trainProfile.loc[index, 'age'])

        labelsTrain.append(age)
    
    featuresTrain = featuresTrain / 255


    # put labels into category vector
    imgAge.setRanges(labelsTrain)
    labelsTrain = np.array(labelsTrain)
    labelsTrain = np_utils.to_categorical(labelsTrain)

    #features should be 9500 different images -> test 2850 (30%) train 6650 (70%)
	# split features and labels into train and test
    X_train = featuresTrain[0:6650, :]  
    X_test  = featuresTrain[6650:, :]
    y_train = labelsTrain[0:6650, :]
    y_test = labelsTrain[6650:, :]

    print("Done processing training dataset!")

    testProfile = pd.read_csv(inputArg + "profile/profile.csv")
    featuresTest = np.empty((testProfile.shape[0], 128, 128, 1))

    for index in range(0, testProfile.shape[0]):
        userid = testProfile.loc[index, 'userid']

        img = load_img(inputArg + "image/" + userid + '.jpg', color_mode="grayscale", target_size=(128, 128)) #size account

        img_numpy_array = img_to_array(img)

        featuresTest[index] = img_numpy_array

    featuresTest = featuresTest / 255

    print("Done processing testing dataset!")

    ageModel = imgAge.age_cnn(4)

    ageModel.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=200)

    predictions = ageModel.predict(featuresTest)
    predictions = np.argmax(predictions, axis=1)

    scores = ageModel.evaluate(X_test, y_test, verbose=0)
    print("GrayScale 128 X 128 CNN Error: %.2f%%" % (100-scores[1]*100))

    age_results = imgAge.getRanges(predictions) 
    #####################################################################################################

    ## now I need to export to xml with the averages.

    #Averages of age, gender, personality traits
    gender, ageRange, opeAvg, conAvg, extAvg, agrAvg, neuAvg = getAverages(trainProfile)

    testProfile = pd.read_csv(inputArg + "profile/profile.csv")

    for x in range(0, testProfile.shape[0]):
        rowId = testProfile.loc[x]['userid']
        createFile(rowId, age_results[x], gender_results[x], str(extAvg), str(neuAvg), str(agrAvg), str(conAvg), str(opeAvg), outputArg) #all calcs need to be cast to strings!









    






