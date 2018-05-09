# -*- coding: utf-8 -*-
"""
=======================IMDB MOVIE REVIEW SENTIMENT ANALYSIS====================
ROLES OF EACH TEAM MEMBER FOR THIS MODEL:
	AVNEET
	- Taken care of word embedding and multi-layered model for project.py

	PARAS
	- Taken care of developing and finding results of one dimensional CNN model
	
PROBLEM:
    The purpose of the project is to determine whether a given movie review has
    a positive or negative sentiment using word embedding and One-Dimensional Convolutional Neural Network.	

DATA: 
    - The dataset is the Large Movie Review Dataset often referred to as the IMDB dataset.
    - It is a collection of 50,000 reviews from IMDB, allowing no more than 30 reviews per movie.
    - Test and Training data have 25000 records each.
	- Dataset contains an even number of positive and negative reviews,so both test and training data have 12500 positive and 12500 negative reviews each.
	- A negative review has a score ≤ 4 out of 10, a positive review has a score ≥ 7 out of 10 and Neutral reviews are not included in the dataset.
	
USAGE:
    The program can be called via a command-line interface.
    
COMMAND FORMAT:
	python project.py 	

EXAMPLE:
	Consider the program has been trained for the below reviews:
	pos - "I dont know why people think this is such a bad movie. Its got a pretty good plot, some good action, and the change of location for Harry does not 
		hurt either. Sure some of its offensive and gratuitous but this is not the only movie like that. Eastwood is in good form as Dirty Harry, and I 		
		liked Pat Hingle in this movie as the small town cop. If you liked DIRTY HARRY, then you should see this one, its a lot better than THE DEAD POOL. 4/5"
	neg - "What happens when an army of wetbacks, towelheads, and Godless Eastern European commies gather their forces south of the border? Gary Busey kicks 
		their butts, of course. Another laughable example of Reagan-era cultural fallout, Bulletproof wastes a decent supporting cast headed by L Q Jones and Thalmus Rasulala."
	
	Then, the classifier should predict the sentiment of the test review:
	"The script for this movie was probably found in a hair-ball recently coughed up by a really old dog. Mostly an amateur film with lame FX. For you Zeta-Jones fanatics: she has the credibility of one Mr. Binks." - neg

ALGORITHM (Step-by-step):
	1. IMBD is an in-built Keras datset Keras allows to load the dataset in 
		a format that is ready for use in neural network and deep learning models
    2. Reviews have been preprocessed, and each review is encoded as a sequence of word indexes 
		(integers).For convenience, words are indexed by overall frequency in the dataset, so that for
		instance the integer "4" encodes the 4th most frequent word in the data. This allows for quick
		filtering operations such as: "only consider the top 1,000 most common words, but eliminate the 
		top 50 most common words.
    3. This is a technique where words are encoded as real-valued vectors in a high-dimensional space, 
	    where the similarity between words in terms of meaning translates to closeness in the vector space.
	    We used a 50-dimension vector to represent each word.
    4. We are only interested in the first 5,000 most used words in the dataset. Therefore we filtered with 5000 words.
    5. The average review has just under 300 words with a standard deviation of just over 200 words. 
	    Hence we choose to cap the maximum review length at 500 words, truncating 
	    reviews longer than that and padding reviews shorter than that with 0 values.
    6. The output of this first layer would be a matrix with the size 50×500 for a
	    given review training or test pattern in integer format.
    7. Embedding layer as the input layer, setting the vocabulary to 5,000,
	    the word vector size to 50 dimensions and the input_length to 500.
	8. We will flatten the Embedded layers output to one dimension, then use 
	    one dense hidden layer of 200 units. The output layer has one neuron and 
	    will use  a sigmoid activation to output values of 0 and 1 as predictions.  
    9. Convolutional neural networks were designed to be used for image data the 
	    same properties that make the CNN model attractive for learning to recognize 
	    objects in images can help to learn structure in paragraphs of words, namely 
	    the techniques invariance to the specific position of features.

ACCURACIES:
For Multilayer Perceptron Layer
      The accuracy achieved is: 87.41%
		
For One Dimensional-CNN
      The accuracy achieved is: 88.81%
	  
REFERENCES:
    [1] https://keras.io/
	[2] https://www.tensorflow.org/tutorials/word2vec

AUTHOR NAME: Avneet Pal Kour & Paras Sethi
DATE: May 8, 2018
"""
#loading the Libraries required using modelling going forward
import numpy
#IMBD is an in-built Keras datset Keras allows to import the dataset 
#in a format that is ready for use in neural network and deep learning models
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# imbd.load_data() will load the IMBD dataset first time to the computer
# and load it in home directory  ~/.keras/datasets/
#code from Keras documentation https://keras.io/datasets/  
# but did not pass any arguments to the function but
#original code had arguments for imbd.load_data() function
(X_train, y_train), (X_test, y_test) = imdb.load_data()
#concatenating test and train predictors and response variables
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print("Training data: ")
#printing the dimensions of data loaded both training and testing data
print(X.shape)
print(y.shape)

#Word embedding
#This is a technique where words are encoded as real-valued vectors
#in a high-dimensional space, where the similarity between words in terms 
#of meaning translates to closeness in the vector space.

#We are only interested in the first 5,000 most used words in the dataset. 
#Therefore we filtered with 5000 words.5000 words only becasuse paper we followed was using 5000 words

#code from Keras documentation https://keras.io/datasets/ 
#only passed nb_words argumen to the function but
#the original code had all arguments that can be passed for imbd.load_data() function
imdb.load_data(nb_words=5000) #load dataset

#word padding
#The average review has just under 300 words with a standard deviation of just over 200 words.
#Hence we choose to cap the maximum review length at 500 words, 
#had to pad because tenserflow needs same lenght units inorder to create a matrix
#code from https://keras.io/preprocessing/sequence/ passed different arguments to the function

X_train =sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

#Creating embedding layer 
#We used a 50-dimension vector to represent each word.
#https://keras.io/layers/embeddings/ the original function embedding has all the arguments
#as it is a tutorial but we passed only 3 arguments https://www.tensorflow.org/tutorials/word2vec mentioned
Embedding(5000, 50, input_length=500)

#Building Multi-Layer Perceptron Model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top n words, zero the rest

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
#We will bound reviews at 500 words, truncating longer reviews and zero-padding shorter 

X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)


# create the sequential model using embedding layer
#different code bits combined from https://keras.io/models/sequential/#the-sequential-model-api 
#accoording to our problem
#used 50 dimensions according to paper
model = Sequential()
model.add(Embedding(5000, 50, input_length=500))

#We will flatten the Embedded layers output to one dimension, 
#then use one dense hidden layer of 200 neurons. 
#The output layer has one neuron and will use a sigmoid activation 
#to output values of 0 and 1 as predictions and finally optimize accuracy using adam from https://keras.io/optimizers/
#done according to example in https://datatables.net/reference/api/flatten() and 
#https://keras.io/models/sequential/#the-sequential-model-api

#Used 200 units for dense layers as their is a thumb rule that we should have as many units in dense layer
#as the legnth of matrix which is 500 here but if its near to 500 it will lead to overfitting
#so we took little less than half as taking more less may lead to under fitting
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
#To fit 200 neurons of dense embedding used epoch as 2  as we had 2 classes so trained the model 
#in 2 eporchs of 105 batch sizes each
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=105, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#One-Dimensional Convolutional Neural Network Model 

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load the dataset but only keep the top n words, zero the rest

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
# pad dataset to a maximum review length in words

X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# create the model one dimensional convolution network
model = Sequential()
model.add(Embedding(5000, 50, input_length=500))
model.add(Conv1D(filters=50, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
#Used 200 units for dense layers as their is a thumb rule that we should have as many units in dense layer
#as the legnth of matrix which is 500 here but if its near to 500 it will lead to overfitting
#so we took little less than half as taking more less may lead to under fitting
model.add(Dense(200, activation='relu'))
#The output layer has one neuron and will use a sigmoid activation 
#to output values of 0 and 1 as predictions and finally optimize accuracy using adam from https://keras.io/optimizers/
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=105, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



