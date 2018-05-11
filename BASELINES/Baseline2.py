"""
Author Name: Paras Sethi
Date: April 29, 2018

USAGE:
    The program can be called via a command-line interface.
    It takes just one arguments i.e. the input file name which is in csv format
    
COMMAND FORMAT:
    python Baseline2.py imdb_master.csv

imdb_master.csv is the input file for our project
This is the baseline model. 
We have analysed what is the basic approach most of the people do for sentiment analysis. 
It includes the below steps:
1: Cleaning the reviews. For this most of the people approach with removing punctuations, stop words, doing stemming.
2: Training the cleaned reviews with NaiveBayes Classifier
3: Applying the classifer results on test dataset
4: Predicting the accuracy and confusion matrix

Results: 
Accuracy: 83.41%
Confusion Matrix
Predicted    	neg    pos    __all__
Actual
neg        	10537  1963   12500
pos         	1964   10536  12500
__all__    	12501  12499  25000
"""

import nltk
import chardet
import pandas as pd
from nltk.stem import WordNetLemmatizer
from pandas_ml import ConfusionMatrix
import argparse 


# TAKING THE FILE INPUTS
parser = argparse.ArgumentParser()
parser.add_argument(dest='file')
args = parser.parse_args()
input = args.file


def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

# Read data from files 
with open(input, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
train=pd.read_csv(input, encoding=result['encoding'])

# Converting the tweets to lowercase
train["review"] = [i.lower() for i in train["review"]]

# Splitiing the data into train and test set and their respective responses
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

# Cleaning the train reviews
reviews=[]
for sent in x_train:
    sent=sent.replace("<br />"," ")
    tokenizer = nltk.WhitespaceTokenizer()
    intermediate = tokenizer.tokenize(sent)
    wordnet_lemmatizer = WordNetLemmatizer()
    intermediate = [wordnet_lemmatizer.lemmatize(i) for i in intermediate]
    stop = nltk.corpus.stopwords.words('english')
    intermediate = [i for i in intermediate if i not in stop]
    reviews.append(intermediate)
 
# Cleaning the test reviews 
testing=[]
for sent in x_test:
    sent=sent.replace("<br />"," ")
    tokenizer = nltk.WhitespaceTokenizer()
    intermediate = tokenizer.tokenize(sent)
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    intermediate = tokenizer.tokenize(sent)
    stop = nltk.corpus.stopwords.words('english')
    intermediate = [i for i in intermediate if i not in stop]
    testing.append(intermediate)
    
training=[]
for i in range(len(reviews)):
    training.append([format_sentence(str(reviews[i])), y_train[i]])
from nltk.classify import NaiveBayesClassifier

# Using NaiveBayes Classifier
classifier = NaiveBayesClassifier.train(training)

classifier.show_most_informative_features()


#print("\nThe result of sentiment analysis is given below:\n")
result = []
for i in testing:
    result.append([format_sentence(str(i)), classifier.classify(format_sentence(str(i)))])

correct=0
wrong=0
y_pred=[]
# Calculating the accuracy
for i in range(len(result)):
    y_pred.append(result[i][1])
    if result[i][1]==y_actu[i]:
        correct+=1
    else:
        wrong+=1

accuracy=correct/(correct+wrong)
print("\nThe accuracy achieved is: "+str("%.2f" % (accuracy*100))+"%\n")
print("\nBelow is the confusion matrix:\n")
# Making the confusion matrix
cm = ConfusionMatrix(y_actu, y_pred)
print(cm)
cm.print_stats()