# MovieReviews

IMDB MOVIE REVIEW SENTIMENT ANALYSIS

PROBLEM:
    The purpose of the project is to determine whether a given movie review has
    a positive or negative sentiment using maximum entropy,random
    forest classification techniques and One-Dimensional Convolutional Neural Network.

DATA: 
    - The dataset is the Large Movie Review Dataset often referred to as the IMDB dataset.
    - It is a collection of 50,000 reviews from IMDB, allowing no more than 30 reviews per movie.
    - Test and Training data have 25000 records each.
	- Dataset contains an even number of positive and negative reviews,so both test and training data have 12500 positive and 12500 negative reviews each.
	- A negative review has a score ≤ 4 out of 10, a positive review has a score ≥ 7 out of 10 and Neutral reviews are not included in the dataset.

PROGRAM:
	We have two source code files i.e. "project.py" and "Project_1.py". Since, we have differnt 
	approach for different models that is why we have created separated files for them.
	project.py - for convolutional neural network
	Project_1.py - for random forest and maximum entropy classifiers

USAGE:
    The program can be called via a command-line interface.
    It takes just one argument i.e. the input file name which is in csv format
    
COMMAND FORMAT:
    python Project_1.py imdb_master.csv   # for random forest and maximum entropy classifiers
	python project.py 					  # for convolutional neural network

imdb_master.csv is the input file for our project and is present in the data directory

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

	FOR RANDOM FOREST AND MAXIMUM ENTROPY CLASSIFIERS:
    1. The program first reads the input file which contains the reviews and
        sentiments for both train and test.
    2. Then we divide the data in input file into train set, train response,
        test set and test response(which will be useful to find accuracy)
    3. Then we proceed with cleaning the data. The first thing we are doing is 
        checking whether any punctuation mark in reviews is an emoticon or not.
        If it is an emoticon, we are replacing it with the sentiment associated
        with that emoticon. we have a method analyze_review() to do so.
    4. cleaning_review() method is used to prerocess the reviews and remove
        any unwanted thing that is not useful. In our reviews, we have a break tag
        </br> which is not necessary, so we are first removing the tag. Then are removing 
		unnecessary puntuation marks that are left after replacing emoticons. 
		Then we convert the words to lower case and split them,
		lemmatizing it and then combining the words to form sentences.
    5. Now we have cleaned data, we are going to convert in into bag of words.
        The Bag of Words model learns a vocabulary from all of the documents, 
        then models each document by counting the number of times each word 
        appears.
    6. At this point, we have numeric training features from the Bag of Words 
        and the original sentiment labels for each feature vector, so we have 
        applied supervised machine learning model i.e. maximum entropy, SVM 
        and random forest. 
    7. Once our model is trained, we apply the results on the test set and 
        predicting the sentiments of the test set.
    8. For our predicted result, we are compairing each sentiment with our
        already given result in the data and calculating the accuracy and
        deriving the confusion matrix.
	
	FOR CONVOLUTIONAL NEURAL NETWORK:
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
        .
ACCURACIES:
For Maximum Entropy Classifier:
    The accuracy achieved is: 87.59%
    Below is the confusion matrix:

        Predicted    neg    pos  __all__
        Actual                          
        neg        10895   1605    12500
        pos         1497  11003    12500
        __all__    12392  12608    25000

For Random Forest: 
    The accuracy achieved is: 84.29%
    Below is the confusion matrix:

        Predicted    neg    pos  __all__
        Actual
        neg        10537   1963    12500
        pos         1964  10536    12500
        __all__    12501  12499    25000
		
For Multilayer Perceptron Layer
      The accuracy achieved is: 87.41%
		
For One Dimensional-CNN
      The accuracy achieved is: 88.81%
	  
REFERENCES:
  [1] https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
	[2] https://keras.io/
	[3] https://www.tensorflow.org/tutorials/word2vec
	[4] https://www.kaggle.com/utathya/sentiment-analysis-of-imdb-reviews/data
  [5] http://ai.stanford.edu/~amaas//papers/wvSent_acl2011.pdf
