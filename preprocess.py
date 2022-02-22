

from cv2 import INTER_AREA, INTER_LINEAR
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


"""
If I'm processing training images I can skip bad profile images that don't resemble faces
"""
"""
If I'm processing test images I need to try to preprocess them, viable or not.
"""
#isTraining is a boolean, if true then I can scrap images that don't resemble faces. 
def preprocess_images(profileData, isTraining):
	viable_faces = [] #faces classified, two eyes classified, rotated and cropped
	unsure_faces = [] #face classified, cropped
	bad_faces = [] #nothing classified, shrinked

	#pre built cv2 classifiers for face and eyes.  There are different xmls I can try for best combo.
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

	#directory of the training images
	mydirectory = 'training/image/' #might change with system arguments ex. sys.argv[0]

	counter = 0
	#for each image in the training folder
	for index in range(0, profileData.shape[0]):
		# Get the image matrix, the id [image matrix, id] #I can use this for train data and for test data.
		userid = profileData.loc[index, 'userid']

		if(isTraining): #if training then I have access to gender and age
			### I should just calculate the gender right here
			gender = profileData.loc[index, 'gender']
			##

			### I should just calculate the gender right here
			age_range = profileData.loc[index, 'age']
			##
		else: #give random val indicator
			gender = -1 #error value
			age_range = -1

		print("Working on " + str(counter))
		counter += 1

		#jpg to grayscale image matrix (unpreprocessed)
		img = cv2.imread(mydirectory + userid + '.jpg', cv2.IMREAD_GRAYSCALE)

		#result is an array containing all the detected faces, as rectangle positions. We can plot it easily
		faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

		#if there is a detectable face
		if(len(faces_detected) >= 1):

			#position and size of face 
			(x, y, w, h) = faces_detected[0]  #somtimes faces_detected will come back empty...
			#cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1) #draws a square around the face


			#identifying the eyes. 
			eyes = eyes_cascade.detectMultiScale(img[y:y+h, x:x+w]) #some images have identifiable eyes and some do not.
			# for (ex, ey, ew, eh) in eyes: #draws rectangles on eyes.
			# 	cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 255), 1) 

			#using only images with 2 eyes detected (could be issues if more eyes are detected)
			if(len(eyes) == 2): 
				crazy_angle = False;
				angle = 0 
				if(eyes[0][0] < eyes[1][0]): #then eye 0 is the left eye.
					leftEye = eyes[0]
					rightEye = eyes[1]
				else:
					rightEye = eyes[0]
					leftEye = eyes[1]

				#calcualte the x and y midpoint for left and right eye.
				leftMidX = leftEye[0] + (leftEye[2] / 2)
				leftMidY = leftEye[1] + (leftEye[3] / 2)
				rightMidX = rightEye[0] + (rightEye[2] / 2)
				rightMidY = rightEye[1] + (rightEye[3] / 2)
				#distances between midpoints
				diffX = abs(leftMidX - rightMidX)
				diffY = abs(leftMidY - rightMidY)
				# diffX = leftMidX - rightMidX
				# diffY = leftMidY - rightMidY

				#atan2 allows for some measurements that are undefined in atan. 
				angle = math.degrees(math.atan2(diffY, diffX))

				#If the angle is greater than 80 degrees then issue with classifier and not good 
				if(abs(angle) >= 82): # I shouldn't rotate an image more than 80% I might even lower this, average is about 7 degrees. 
					crazy_angle = True

			
				# STRAIGHTEN FACE, once we've detected both eyes by calculating the angle between them. 
				rows, cols = img.shape[:2]
				M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
				img_rotated = cv2.warpAffine(img, M, (cols, rows)) #img_rotated is the rotated image.
				##testing
				#plt.imshow(cv2.cvtColor(img_rotated, cv2.IMREAD_GRAYSCALE))

				#p = 10 #padding 

				#Cropped to show only face. 
				# img_cropped =  img_rotated[y-p+1:y+h+p, x-p+1:x+w+p]
				img_cropped =  img_rotated[y:y+h, x:x+w]
				#plt.imshow(cv2.cvtColor(img_cropped, cv2.IMREAD_GRAYSCALE))

				#resize to ml training size
				img_trainable = cv2.resize(img_cropped, (128, 128), interpolation=INTER_LINEAR)

				if(crazy_angle): #wasn't a viable face after all... #this should already be cropped
					img_cropped = img[y:y+h, x:x+w] #just include classified face 
					img_resize = cv2.resize(img_cropped, (128, 128), interpolation=INTER_LINEAR) #image is not cropped and needs to be shrinked
					#plt.imshow(cv2.cvtColor(img_resize, cv2.IMREAD_GRAYSCALE))
						
					unsure_faces.append([img_resize, userid, gender, age_range]) #don't rotate just crop

				else:		#ADD TO VIABLE FACES ARRAY
					viable_faces.append([img_trainable, userid, gender, age_range])
					#plt.imshow(cv2.cvtColor(img_trainable, cv2.IMREAD_GRAYSCALE))
			
			else:
				#faces are still recognized, just not two eyes, semi good training.
				img_cropped =  img[y:y+h, x:x+w]
				img_resize = cv2.resize(img_cropped, (128, 128), interpolation=INTER_LINEAR) #image is cropped just not rotated (only face detected)
				plt.imshow(cv2.cvtColor(img_resize, cv2.IMREAD_GRAYSCALE))
				unsure_faces.append([img_resize, userid, gender, age_range])
		else:
			img_resize = cv2.resize(img, (128, 128), interpolation=INTER_AREA) #image is not cropped/classified and needs to be shrinked from original/thrown away
			bad_faces.append([img_resize, userid, gender, age_range])

	if(isTraining):
		return viable_faces# + unsure_faces #train on only good photos?
	else:
		return viable_faces + unsure_faces + bad_faces



#bad faces 1297
# unsure_faces 5376
# viable_faces 2827


			



if __name__== "__main__":
	mydirectory = 'training/image/' #possible place for sys.arg[0]

	profileData = pd.read_csv('training/profile/profile.csv')

	trainable = preprocess_images(profileData, True)
	print(len(trainable)) #then length of good training images is 2825/9000

	#Storing so that I don't have to wait the 4 minutes to preprocess every time
	for i in range (0, len(trainable)):
		cv2.imwrite('imgProcessedGood/' + str(trainable[i][1]) + '.jpg', trainable[i][0])

#age good faces -> 34% error
#value good gaces -> 20% error
#hopefully this will work?