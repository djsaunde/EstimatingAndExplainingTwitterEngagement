'''
Script which contains helper functions for the main script(s).

author: Dan Saunders (djsaunde@umass.edu)
'''

from __future__ import division

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer

from nltk.tokenize import TweetTokenizer

from collections import defaultdict

import numpy as np

import csv


def remove_retweets(tweets):
    '''
    Function to remove tweets that are retweeted from another Twitter handle.
    For now, this simply checks for the occurrence of 'RT'. I believe there is
    a more clever way to do this (check Twitter API).
    
    input:
        tweets: list of tweets (typically 200, sometimes less).
    
    output:
        list of tweets with retweets removed.
    '''
    
    # maintain a list of normal tweets (not retweets, that is)
    not_retweets = []
    # loop through each tweet
    for tweet in tweets:
        # if the tweet doesn't contain the string literal 'RT'...
        if 'RT' not in tweet.text:
            # add it to the returned list
            not_retweets.append(tweet)
            
    return not_retweets
    
    
def parse_dataset(dataset, regression_value='likes'):
    '''
    Function which reduces a dataset of tweets into its constituent feature 
    representation. This is a first example; there are many possible representations
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # open the data file associated with the handle passed in as a parameter
    with open('../data/' + dataset, 'rb') as f:
        tweets_list = list(csv.reader(f))
    
    # store the tweet datetime in an array
    # datetimes = np.array([ tweets_list[i][0] for i in range(len(tweets_list)) ])
    
    # create a TweetTokenizer object, courtesy of NLTK
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True)
    
    # instantiate a count vectorizer to vectorize the dataset of tweet text
    vectorizer = CountVectorizer(min_df=1   )
    # get a list of tweet text
    tweets_text = np.asarray([ tweets_list[i][1] for i in range(len(tweets_list)) ], dtype=str)
    
    # fit / transform the dataset using the count vectorizer
    tweets_transformed = vectorizer.fit_transform(tweets_text)
    
    if regression_value == 'likes':
        # store the number of likes associated with each tweet in an array
        likes = np.asarray([ tweets_list[i][2] for i in range(len(tweets_list)) ], dtype=np.int64)
        # return tweets text and likes
        return (tweets_transformed, likes)
        
    elif regression_value == 'retweets':
        # store the number of retweets associated with each tweet in an array
        retweets = np.asarray([ tweets_list[i][3] for i in range(len(tweets_list)) ], dtype=np.int64)
        # return tweets text and retweets
        return (tweets_transformed, retweets)
        
    else:
        raise NotImplementedError
        
        
def parse_dataset2(dataset, regression_value='likes'):
    '''
    Function which reduces a dataset of tweets into its constituent feature 
    representation. This is a first example; there are many possible representations
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # open the data file associated with the handle passed in as a parameter
    with open('../data/' + dataset, 'rb') as f:
        tweets_list = list(csv.reader(f))
    
    # store the tweet datetime in an array
    # datetimes = np.array([ tweets_list[i][0] for i in range(len(tweets_list)) ])
    
    # create a hashing vectorizer to transform the tweets into hash vector representation
    hv = HashingVectorizer(n_features=2**18)
    # get a list of tweet text
    tweets_text = [ tweets_list[i][1] for i in range(len(tweets_list)) ]
    
    # use the hashing vectorizer to transform the raw text into hash vectors
    tweets_transformed = hv.transform(tweets_text).toarray().astype(np.int32)
    
    if regression_value == 'likes':
        # store the number of likes associated with each tweet in an array
        likes = np.array([ tweets_list[i][2] for i in range(len(tweets_list)) ])
        # return tweets text and likes
        return (tweets_transformed, likes)
        
    elif regression_value == 'retweets':
        # store the number of retweets associated with each tweet in an array
        retweets = np.array([ tweets_list[i][3] for i in range(len(tweets_list)) ])
        # return tweets text and retweets
        return (tweets_transformed, retweets)
        
    else:
        raise NotImplementedError
    

def split_dataset(dataset, train_perc, test_perc):
    '''
    This function splits the dataset into a training and test set, as per user
    inputer parameters.
    
    input:
        dataset: datapoints and regression labels of the full parsed datset
        train_perc: percentage of dataset to use for training our model
        test_perc: percentage of dataset to use for testing our model
        
        note -> train_perc + test_perc
        
    output:
        two slices of the dataset, split by percentage parameters passed in
    '''
    
    # check to make sure percentages add to 100%
    assert train_perc + test_perc == 100
    
    # find the split, cast it to an integer
    split = int(len(dataset[1]) * (0.01 * train_perc))
    
    # return slices of the dataset based on the calculated split
    return (dataset[0][:split], dataset[1][:split]), (dataset[0][split:], dataset[1][:split])
    

def build_model(train_data, model_type, cross_validate=False):
    '''
    This function fits an estimator (of the type model) to the training dataset
    (train_data).
    
    input:
        training_set: the set of training data from which to build our model
        model_type: the scikit-learn model of choice given by user input parameter
        
    output:
        trained model fit to the training dataset
    '''
    
    model = None
    
    if model_type == 'linear regression':
        # create linear regression model
        model = LinearRegression()
        
        if cross_validate:
            # cross-validate the model 
            model = cross_validate(train_data, model_type, model)
            model.fit(train_data[0], train_data[1])
        
        else:
            # use default parameters
            model.fit(train_data[0], train_data[1])
        
    elif model_type == 'support vector regression':
        # create support vector regression model
        model = SVR()
        
        if cross_validate:
            # cross-validate the model 
            model = cross_validate(train_data, model_type, model)
            model.fit(train_data[0], train_data[1])
        
        else:
            # use default parameters
            model.fit(train_data[0], train_data[1])
        
    elif model == 'neural network regression':
        # create neural network regression model
        model = MLPRegressor()
        
        if cross_validate:
            # cross-validate the model 
            model = cross_validate(train_data, model)
            model.fit(train_data[0], train_data[1])
        
        else:
            # use default parameters
            model.fit(train_data[0], train_data[1])
        
    else:
        raise NotImplementedError
    
    # return the trained / cross-validated and trained model    
    return model
    
    
def get_score(model, test_data):
    '''
    This function returns the score of the trained model on the test datset.
    
    input:
        model: the regression model which is fit to the training data
        test_set: the set of test data which we use to test predictive capacity

        
    output:
        score of the trained model on the test set
    '''
    
    return model.score(test_data[0], test_data[1])



