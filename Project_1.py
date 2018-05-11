# -*- coding: utf-8 -*-
"""
==================IMDB MOVIE REVIEW SENTIMENT ANALYSIS========================

ROLES OF EACH TEAM MEMBER FOR THIS PROGRAM:
	AVNEET
	- Splitting the input file into train and test 
	- Implementing the bag of words
	- Implementing and finding results of random forest model

	PARAS
	- Cleaning of text
	- Implementing and finding results of maximum entropy model
    
PROBLEM:
    The purpose of the project is to determine whether a given movie review has
    a positive or negative sentiment using maximum entropy, and random
    forest classification techniques.

DATA: 
    - The dataset is the Large Movie Review Dataset often referred to as the 
        IMDB dataset.
    - It is a collection of 50,000 reviews from IMDB, allowing no more than 30 
        reviews per movie.
    - Test and Training data have 25000 records each.

USAGE:
    The program can be called via a command-line interface.
    It takes just one arguments i.e. the input file name which is in csv format
    
COMMAND FORMAT:
    python Project_1.py imdb_master.csv

imdb_master.csv is the input file for our project

ALGORITHM (Step-by-step):
    1. The program first reads the input file which contains the review and
        sentiments for both train and test
    2. Then we divide the data in input file into train set, train response,
        test set and test response(which will be useful to find accuracy)
    3. Then we proceed with cleaning the data. The first thing we are doing is 
        checking whether any punctuation mark in reviews is an emoticon or not.
        If it is an emoticon, we are replacing it with the sentiment associated
        with that emoticon. we have a method analyze_review() to do so.
    4. cleaning_review() method is used to prerocess the reviews and remove
        any unwanted thing that is not useful. In our reviews, we have a tag
        </br> which is not necessary, so we are first removing the tag. Then we
        replace the emoticons with words and then we are removing the unnecessary,
        puntuation marks. Then we are splitting,converting the sentences to 
        lowercase, lemmatizing it and then combining the words to form sentences.
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

Note: We have tried to create the model a bit interactive by giving it a name
    "ANALYZER" and printed the lines as if the model is talking with you.
    
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
    
REFERENCES:
    [1] https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
    [2] https://www.kaggle.com/utathya/sentiment-analysis-of-imdb-reviews/data
    
AUTHOR NAME: Avneet Pal Kour & Paras Sethi
DATE: May 8, 2018
"""
import chardet
import pandas as pd
from nltk.stem import WordNetLemmatizer
from pandas_ml import ConfusionMatrix
import re
from sklearn.feature_extraction.text import CountVectorizer
import argparse 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import time #to calculate the execution time
start = time.clock() # start of execution


## TAKING THE FILE INPUTS
#parser = argparse.ArgumentParser()
#parser.add_argument(dest='file')
#args = parser.parse_args()
#input = args.file

print("Hi!!! \nWelcome!!!")
print("\nMy name is ANALYZER.\nI analyze the sentiments of IMDB movie reviews\n")
print("\nNow I am reading the input file. This is going to take some time, so please bear with me")
# Read data from files 
with open("imdb_master.csv", 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
train=pd.read_csv("imdb_master.csv", encoding=result['encoding'])
print("\nI have read the file. Now dividing it into train and test sets:\n")


######### Taken care by Avneet Pal ##########
# Splitting the data into train and test set according to the type in data
x_train=[]
y_train=[]
x_test=[]
y_actu=[]
for i in range(len(train)):
    if train["type"][i]=="train":
        x_train.append(train["review"][i])
        y_train.append(train["label"][i])
    else:
        x_test.append(train["review"][i])
        y_actu.append(train["label"][i])



######### Taken care by Paras ##########
# To analyse the puntuations and replacing them with the associated words,
# We have taken the code from a github repository
# https://github.com/aritter/twitter_nlp/blob/master/python/emoticons.py
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')
NormalEyes = r'[8:=]'
Wink = r'[;*]'
NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...
HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned
Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes+ NoseArea + SadMouths )
Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)

Emoticon = (
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea + 
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"|"+HappyMouths+")"
)
Emoticon_RE = mycompile(Emoticon)
#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)
# Till this part, we have taken the work from the above mentioned URL

# Below is the function that we have made using the above code
def analyze_review(text):
    text=re.sub(Happy_RE, r'Happy', text)
    text=re.sub(Sad_RE, r'Sad', text)
    text=re.sub(Wink_RE, r'Wink', text)
    text=re.sub(Tongue_RE, r'Tongue', text)
    return text


######### Taken care by Paras ##########
# Function to convert a reviews to a sequence of words,
# optionally removing stop words.  Returns a list of words.
def cleaning_review(review):
    # 1. Remove HTML tag "<br />"
    review_text = review.replace("<br />"," ")
    # 2. Replace emoticons with words
    review_text= analyze_review(review_text)
    # 3. Remove Non letters and tokenize
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # 4. Convert words to lower case and split them
    review_text = review_text.lower().split()
    # 5. Lemmatizing the data
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(i) for i in review_text]
    # 6. Joining the words to form sentences
    return(" ".join(words))


######### Taken care by Paras ##########
# Cleaning the train and test reviews and saving them in training and testing 
training=[]
testing=[]
for i in range(len(x_train)):
    training.append(cleaning_review(x_train[i]))
    testing.append(cleaning_review(x_test[i]))


######### Taken care by Avneet Pal ##########
print ("\nCreating the bag of words...\n")
# Initialize bag of words tool from scikit-learn's
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,\
                             preprocessor = None,stop_words = None,max_features = 5000) 
# fit_transform() does two functions: 
# First, it fits the model and learns the vocabulary; 
# second, it transforms our training data into feature vectors. 
# The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(training)
# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(testing)
test_data_features = test_data_features.toarray()
print("\nBag of words is created.")

print("\nNow, I'm starting with the training of classifiers\n")
######### Taken care by Paras ##########
###################
# Maximum Entropy #
###################
print ("\nTraining the maximum entropy classifier...")
# We'll use LogisticRegressionCV
# As, Logistic Regression CV aka MaxEnt classifier
# Fit the model to the training set, using the bag of words
maxent = LogisticRegression(solver='sag',\
                              max_iter=100,\
                              fit_intercept=True)
maxent=maxent.fit( train_data_features, y_train)
# Use the random forest to make sentiment label predictions
result_maxent = maxent.predict(test_data_features)
print("Maximum Entropy Classifier is trained.\n")

######### Taken care by Avneet Pal ##########
####################
##  Random Forest  #
####################
print ("Training the random forest tree...")
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 
# Fit the forest to the training set, using the bag of words as features 
forest = forest.fit( train_data_features, y_train)
# Use the random forest to make sentiment label predictions
result_rf = forest.predict(test_data_features)
print("Random forest classifer is trained.\n")

######### Taken care by Paras ##########
# Calculating the accuracy and making confusion matrix
def calculate_accuracy(result):
    correct=0
    wrong=0
    # Calculating the accuracy
    for i in range(len(result)):
        if result[i]==y_actu[i]:
            correct+=1
        else:
            wrong+=1

    accuracy=correct/(correct+wrong)
    print("\nThe accuracy achieved is: "+str("%.2f" % (accuracy*100))+"%\n")
    print("Below is the confusion matrix:\n")
    cm = ConfusionMatrix(y_actu, result)
    print(cm)
    cm.print_stats()

print("\nBelow are the results of random forest:")
calculate_accuracy(result_rf)
print("\nBelow are the results of Maximum Entropy Classifier:")
calculate_accuracy(result_maxent)

end = time.clock() # end of execution
# Print the execution time
print("\nTotal Execution time for all the models is %.2f" %(((end-start)*100)/3600) + " minutes")