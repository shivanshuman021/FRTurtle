import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate,  BatchNormalization, AveragePooling2D , MaxPool2D , Concatenate , Lambda, Flatten, Dense , Layer
from tensorflow.keras import Model
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from fr_utils import *
from inception_blocks import *

from support import *
import time as t


#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

np.set_printoptions(threshold=2**31 - 1)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

#print("Total Params:", FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.math.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.math.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.math.add(tf.subtract(pos_dist,neg_dist),alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.math.reduce_sum(tf.maximum(basic_loss,0))
    ### END CODE HERE ###

    return loss


FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

print("\nmodel loaded ....\n")

database = {}
database["Anshuman"] = img_to_encoding("dataset/train/me.jpeg", FRmodel)
database["Aaditya"] = img_to_encoding("dataset/train/aaditya.jpeg", FRmodel)
database["Aditya"] = img_to_encoding("dataset/train/aditya.jpeg", FRmodel)
database["Chinmay"] = img_to_encoding("dataset/train/chinmay.jpeg", FRmodel)
database["Chaitanya"] = img_to_encoding("dataset/train/chaitanya.jpeg", FRmodel)
database["Chopra"] = img_to_encoding("dataset/train/chopra.jpeg", FRmodel)
database["Kaiwalya"] = img_to_encoding("dataset/train/kaiwalya.jpeg", FRmodel)
database["Parthiv"] = img_to_encoding("dataset/train/parthiv.jpeg", FRmodel)
database["Priyanshu"] = img_to_encoding("dataset/train/priyanshu.jpeg", FRmodel)
database["Soumyadip"] = img_to_encoding("dataset/train/soumyadip.jpeg", FRmodel)
database["Sufiyan"] = img_to_encoding("dataset/train/sufiyan.jpeg", FRmodel)

print("\nDatabase loaded ...\n")

def writer(n):
	str_py = n.upper()
	'''
	val=0
	
	while(val!=1):
		val=1
		str_py=input("Enter A Name : ").upper()
		for i in str_py:
			if(ord(i)<65 or ord(i)>90) and ord(i)!=32:
				val=0
		if val==0:
			print('\nPlease Enter A Valid Name')
	'''

	#ff=int(input("Enter Font Size:\n 1 for 120\n 2 for 60\n 3 for 40\n"))
	#while(ff not in range(1,4)):
	#ff=int(input("\nPlease Choose Correct Options.\nEnter Font Size:\n 1 for 120\n 2 for 60\n 3 for 40\n"))

	ff = 3	
	#pensize_var=int(input("Input pen thickness (2-6):"))
	#while(pensize_var not in range(2,7)):
	#pensize_var=int(input("\nPlease Enter Between 2 and 6\nInput pen thickness (2-6):"))

	pensize_var = 4	
	print('Maximise The New Window')

	name(pensize_var,ff)
	l=0
	for i in range(len(str_py)):
		if (i%(11*ff)==0 and i>0):
			l=l+1
		tab(120*(i%(11*ff))/ff-700,200+120-120/ff-(l*200)/ff)
		if i==0:
			t.sleep(1)
		if str_py[i]==' ':
			continue
		eval(str_py[i]+'()')   	
	t.sleep(2)


def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    ### START CODE HERE ###
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist<0.7:
        writer(str(identity))
        door_open = True
    else:
        writer("not in database")
        door_open = False
        
    ### END CODE HERE ###
        
    return dist, door_open





def who_is_it(image_path, database, model):
    """
    Implements face recognition for the office by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.7:
        writer("not in database")
    else:
        writer(str(identity))
        
    #return min_dist, identity


print("\nexecuting main...\n")

for i in range(15):
	print("Show me your face")
	images_path = str(input())
	who_is_it(images_path, database, FRmodel)


