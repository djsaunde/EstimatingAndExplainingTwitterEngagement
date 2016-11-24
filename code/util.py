'''
Script which contains helper functions for the main script(s).

author: Dan Saunders (djsaunde@umass.edu)
'''

from __future__ import division

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import mean_absolute_error

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
    
    
def get_features_count_vectorizer(tweets_text, regression_value):
    '''
    Function which reduces a dataset of tweets into a feature vectorizer / occurrence counter.
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # instantiate a count vectorizer to vectorize the dataset of tweet text
    vectorizer = CountVectorizer(min_df=1)
    
    # fit / transform the dataset using the count vectorizer
    features = vectorizer.fit_transform(tweets_text)
    
    # return the features we've extracted
    return features
        
        
def get_features_tfidf(tweets_text, regression_value):
    '''
    Function which reduces a dataset of tweets into count vectorizer representation,
    then weight these occurrences by their tf-idf index.
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # instantiate a count vectorizer to vectorize the dataset of tweet text
    vectorizer = CountVectorizer(min_df=1)
    
    # fit / transform the tweets using the count vectorizer
    counts = vectorizer.fit_transform(tweets_text)
    
    # instantiate a tfidf transformer object
    tfidf_transformer = TfidfTransformer()
    
    # fit it to our count vectorized data and then transform said data
    features = tfidf_transformer.fit_transform(counts)
    
    # return the features we've extracted
    return features
        
        
def get_features_hashing_vectorizer(tweets_text, regression_value):
    '''
    Function which reduces a dataset of tweets into a HashingVectorizer object with
    2^14 features. 
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # create a hashing vectorizer to transform the tweets into hash vector representation
    hv = HashingVectorizer(n_features=2**10)
    
    # use the hashing vectorizer to transform the raw text into hash vectors
    features = hv.transform(tweets_text).toarray().astype(np.int32)
    
    # return the features we've extracted
    return features
        
        
def get_features(filename, regression_value, extr_method):
    '''
    Delegate function to feature extraction in order to remove duplicate code
    
    input:
        filename: filename which 
    
    output:
        data processed the way the user specified with the 'extr_method' parameter
    '''
    
    # open the data file associated with the handle passed in as a parameter
    with open('../data/' + filename, 'rb') as f:
        tweets_list = list(csv.reader(f))
        
    # get a list of tweet text
    tweets_text = [ tweets_list[i][1] for i in range(len(tweets_list)) ]
    
    # delegate to feature extraction functions based on 'extr_method' parameter
    if extr_method == 'count vectorizer':
        features = get_features_count_vectorizer(tweets_text, regression_value)
    elif extr_method == 'tfidf':
        features = get_features_tfidf(tweets_text, regression_value)
    elif extr_method == 'hashing vectorizer':
        features = get_features_hashing_vectorizer(tweets_text, regression_value)
    else:
        raise NotImplementedError
        
    # branch based on regression value
    if regression_value == 'likes':
        # store the number of likes associated with each tweet in an array
        values = np.array([ tweets_list[i][2] for i in range(len(tweets_list)) ], dtype=np.int64)
        # return tweets text and likes
        return (features, values)
        
    elif regression_value == 'retweets':
        # store the number of retweets associated with each tweet in an array
        values = np.array([ tweets_list[i][3] for i in range(len(tweets_list)) ], dtype=np.int64)
        # return tweets text and retweets
        return (features, values)
        
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
    return (dataset[0][:split], dataset[1][:split]), (dataset[0][split:], dataset[1][split:])
    

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
        
    elif model_type == 'neural network regression':
        # create neural network regression model
        model = MLPRegressor(early_stopping=True)
        
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
    This function returns the score of the trained model on the test dataset.
    
    input:
        model: the regression model which is fit to the training data
        test_data: the set of test data which we use to test predictive capacity
        
    output:
        score of the trained model on the test dataset
    '''
    
    # return the score of the trained model on the test data
    return model.score(test_data[0], test_data[1])
    
    
def get_mean_abs_error(model, test_data):
    '''
    This function returns the mean absolute error of the trained model on the test dataset.
    
    input:
        model: the regression model which is fit to the training data
        test_data: the set of test data which we use to ttest predictive capacity
        
    output:
        mean absolute error of the trained model on the test dataset
    '''
    
    # get true values, predicted labels
    true_values = test_data[1]
    pred_values = model.predict(test_data[0])
    
    # return mean absolute error of the trained model on the test data
    return mean_absolute_error(true_values, pred_values)
    
