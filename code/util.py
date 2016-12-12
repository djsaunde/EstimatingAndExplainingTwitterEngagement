'''
Script which contains helper functions for the main script(s).

author: Dan Saunders (djsaunde@umass.edu)
'''


# want division of integers to caste to floats
from __future__ import division

# scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

# standard imports
import random, csv, os, re, numpy as np

# matplotlib for plotting
import matplotlib.pyplot as plt


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
            # remove links from tweet text
            tweet.text = remove_link(tweet.text)
            # add it to the returned list
            not_retweets.append(tweet)
            
    return not_retweets
    
    
def get_features_count_vectorizer(tweets_text):
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
    return features, vectorizer
        
        
def get_features_tfidf(tweets_text):
    '''
    Function which reduces a dataset of tweets into count vectorizer representation,
    then weight these occurrences by their tf-idf index.
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # instantiate a tfidf transformer object
    tfidf_transformer = TfidfVectorizer()
    
    # fit it to our count vectorized data and then transform said data
    features = tfidf_transformer.fit_transform(tweets_text)
    
    # return the features we've extracted
    return features, tfidf_transformer
        
        
def get_features_hash_vectorizer(tweets_text):
    '''
    Function which reduces a dataset of tweets into a HashingVectorizer object with
    2^14 features. 
    
    input:
        dataset: filename of the csv file contains tweet data
    
    output:
        datapoints (with some chosen feature representation) and regression labels 
    '''
    
    # create a hashing vectorizer to transform the tweets into hash vector representation
    hv = HashingVectorizer(n_features=2**11)
    
    # use the hashing vectorizer to transform the raw text into hash vectors
    features = hv.fit_transform(tweets_text).toarray().astype(np.int32)
    
    # return the features we've extracted
    return features, hv
        
        
def get_features_and_targets_regression(filename, target_value, extr_method):
    '''
    Delegate function to feature extraction in order to remove duplicate code (for 
    regression task)
    
    input:
        filename: name of .csv file to get dataset from
        target_value: value which we wish to predict via regression
        extr_method: feature extraction method to use on the Twitter dataset
    
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
        features, feature_extractor = get_features_count_vectorizer(tweets_text)
    elif extr_method == 'tfidf':
        features, feature_extractor = get_features_tfidf(tweets_text)
    elif extr_method == 'hash vectorizer':
        features, feature_extractor = get_features_hash_vectorizer(tweets_text)
    else:
        raise NotImplementedError
        
    # branch based on regression value
    if target_value == 'likes':
        # store the number of likes associated with each tweet in an array
        targets = np.array([ tweets_list[i][2] for i in range(len(tweets_list)) ], dtype=np.int64)
        
        # return tweets features and likes
        return features, targets, feature_extractor
        
    elif target_value == 'retweets':
        # store the number of retweets associated with each tweet in an array
        targets = np.array([ tweets_list[i][3] for i in range(len(tweets_list)) ], dtype=np.int64)
        
        # return tweets features and retweets
        return features, targets, feature_extractor
        
    elif target_value == 'both':
        # store the number of likes and retweets associated with each tweet in an array
        likes = [tweets_list[i][2] for i in range(len(tweets_list))]
        retweets = [tweets_list[i][3] for i in range(len(tweets_list))]
        targets = np.array(zip(likes, retweets), dtype=np.int64)
        
        # return tweets features and likes, retweets targets
        return features, targets, feature_extractor
        
    else:
        raise NotImplementedError
        
        
def get_features_and_targets_clustering(filename, extr_method):
    '''
    Delegate function to feature extraction in order to remove duplicate code (for 
    clustering task)
    
    input:
        filename: name of .csv file to get dataset from
        extr_method: feature extraction method to use on the Twitter dataset
    
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
        features, feature_extractor = get_features_count_vectorizer(tweets_text)
    elif extr_method == 'tfidf':
        features, feature_extractor = get_features_tfidf(tweets_text)
    elif extr_method == 'hash vectorizer':
        features, feature_extractor = get_features_hash_vectorizer(tweets_text)
    else:
        raise NotImplementedError   
    
    # store the number of likes + retweets associated with each tweet in an array
    likes = [tweets_list[i][2] for i in range(len(tweets_list))]
    retweets = [tweets_list[i][3] for i in range(len(tweets_list))]
    targets = np.array(likes, dtype=np.int64) + np.array(retweets, dtype=np.int64)
        
    return features, targets, feature_extractor
    
    
def get_features_and_targets_dimensionality_reduction(filename, extr_method):
    '''
    Delegate function to feature extraction in order to remove duplicate code (for 
    dimensionality reduction task)
    
    input:
        filename: name of .csv file to get dataset from
        target_value: value which we wish to predict via regression
        extr_method: feature extraction method to use on the Twitter dataset
    
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
        features, feature_extractor = get_features_count_vectorizer(tweets_text)
    elif extr_method == 'tfidf':
        features, feature_extractor = get_features_tfidf(tweets_text)
    elif extr_method == 'hash vectorizer':
        features, feature_extractor = get_features_hash_vectorizer(tweets_text)
    else:
        raise NotImplementedError
    
    # store the number of likes + retweets associated with each tweet in an array
    likes = [tweets_list[i][2] for i in range(len(tweets_list))]
    retweets = [tweets_list[i][3] for i in range(len(tweets_list))]
    targets = np.array(likes, dtype=np.int64) + np.array(retweets, dtype=np.int64)
        
    return features, targets, feature_extractor
        
        
def unison_shuffled_copies(list1, list2):
    '''
    This function shuffles two lists with a single permutation / shuffling.
    
    input:
        list1: first list to permute / shuffle
        list2: second list to permute / shuffle
        
    output:
        tuple of both lists, shuffled
    '''
    
    # if we are doing multiple regression (likes and tweets)
    if list2.shape[0] == 2:
        # get a permutation on the number of elements in the lists
        p = np.random.permutation(list1.shape[0])
        # use the permutation to permute the lists
        list2[0], list2[1] = list2[0][p], list2[1][p]
        # return the lists
        return list1[p], list2
        
    # get a permutation on the number of elements in the lists
    p = np.random.permutation(list1.shape[0])
    # use the permutation to permute the lists
    return list1[p], list2[p]
    

def split_dataset(dataset, train_perc, test_perc):
    '''
    This function randomizes the order of the dataset, then splits it into a 
    training and test set, as per user input parameters.
    
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
    
    # randomize order of dataset (accounts for time-shift)
    shuffled_dataset = unison_shuffled_copies(dataset[0], dataset[1])
    dataset = shuffled_dataset
    
    # find the split, cast it to an integer
    split = int((dataset[0].shape[0]) * (0.01 * train_perc))
    
    # return slices of the dataset based on the calculated split
    return (dataset[0][:split], dataset[1][:split]), (dataset[0][split:], dataset[1][split:])
    

def build_regression_model(train_data, model_type, cross_validate, num_iters, params):
    '''
    This function fits an estimator (of the type model) to the training dataset
    (train_data).
    
    input:
        training_set: the set of training data from which to build our model
        model_type: the scikit-learn model of choice given by user input parameter
        cross_validate: whether or not to perform random search cross-validation over
            pre-specified parameter distributions
        num_iters: the number of iterations to perform the cross-validation
        
    output:
        trained model fit to the training dataset
    '''
    
    model = None
    
    if model_type == 'linear regression':
        # create linear regression model
        model = LinearRegression()
        
        # NOTE: not any important hyperparameters to tune (in my opinion)!
        
        # use default parameters
        model.fit(train_data[0], train_data[1])
        
    elif model_type == 'support vector regression':
        # create support vector regression model (default to radial basis function kernel)
        model = SVR(kernel='rbf')
        
        if cross_validate:
            # create parameter distributions
            param_distro = {
                'kernel' : [ 'rbf', 'sigmoid' ],
                'C' : [ 0.01, 0.1, 1.0, 10.0 ],
                'epsilon' : [ 0.01, 0.05, 0.1, 0.5, 1.0 ]
            }
            
            # create random grid search object
            model = RandomizedSearchCV(estimator=model, param_distributions=param_distro, n_iter=num_iters, n_jobs=-1, verbose=True)
            
            print '\n', '... performing cross-validation', '\n'
            
            # cross-validate the model
            model.fit(train_data[0], train_data[1])
        
        else:
            # use default parameters
            model.fit(train_data[0], train_data[1])
            
    elif model_type == 'decision tree regression':
        # create decision tree regression model
        model = DecisionTreeRegressor()
        
        if cross_validate:
            # create parameter distributions
            param_distro = {
                'splitter' : [ 'best', 'random' ],
                'min_samples_split' : [ 2, 5, 10, 25, 50 ],
                'min_samples_leaf' : [ 1, 5, 10, 25 ]
            }
            
            # create random grid search object
            model = RandomizedSearchCV(estimator=model, param_distributions=param_distro, n_iter=num_iters, n_jobs=-1, verbose=True)
            
            print '\n', '... performing cross-validation', '\n'
            
            # cross-validate the model
            model.fit(train_data[0], train_data[1])
            
        else:
            # use default parameters
            model.fit(train_data[0], train_data[1])
        
    elif model_type == 'neural network regression':
        # create neural network regression model
        model = MLPRegressor(hidden_layer_sizes=params['layer_sizes'], early_stopping=True, verbose=True)
        
        if cross_validate:
            # create parameter distributions
            param_distro = {
                'hidden_layer_sizes' : [ (2048), (1024), (2048, 1024), (1024, 1024), (1024, 512), ],
                'alpha' : [ 0.00001, 0.0001, 0.001 ],
            }
            
            # create random grid search object
            model = RandomizedSearchCV(estimator=model, param_distributions=param_distro, n_iter=num_iters, n_jobs=6, verbose=True)
            
            print '\n', '... performing cross-validation', '\n'
            
            # cross-validate the model
            model.fit(train_data[0], train_data[1])
        
        else:
            # use default parameters
            model.fit(train_data[0], train_data[1])
        
    else:
        raise NotImplementedError
    
    # return the trained / cross-validated and trained model    
    return model
    
    
def build_clustering_model(data, model_type, cross_validate=False, num_iters=10, num_clusters=10):
    '''
    This function fits a clustering model (of the type model_type) to the given features
    
    input:
        data: the set of features of the data from which to build our model
        model_type: the scikit-learn model of choice given by user input parameter
        cross_validate: whether or not to perform random search cross-validation over
            pre-specified parameter distributions
        num_iters: the number of iterations to perform the cross-validation
        num_clusters: the number of clusters which we consider in fitting the model
        
    output:
        trained model fit to the features of the data
    '''
    
    # create a model variable
    model = None
    
    if model_type == 'agglomerative clustering':
        # instantiate model with default hyperparameter settings
        model = AgglomerativeClustering(n_clusters=num_clusters)
        
        # no scoring function implies no cross validation!
        model.fit(data)
        
        return model
        
    elif model_type == 'kmeans':
        # instantiate model with default hyperparameter settings
        model = KMeans(n_clusters=num_clusters)
        
        if cross_validate:
            # create parameter distributions
            param_distro = {
                    'n_clusters' : range(10, 106, 5) 
                }
            
            # create random grid search object
            model = RandomizedSearchCV(estimator=model, param_distributions=param_distro, n_iter=num_iters, n_jobs=-1, verbose=True)
            
            print '\n', '... performing cross-validation', '\n'
            
            # cross-validate the model
            model.fit(data)
            
            # return the best estimator from the cross-validation search
            return model.best_estimator_
            
        else:
            # use default parameters
            model.fit(data)
            
            # return the model
            return model
            
    else:
        raise NotImplementedError
    
    
def build_dimensionality_reduction_model(data, model_type, cross_validate=False, num_iters=10):
    '''
    This function fits a dimensionality reduction model (of the type model_type) to the given features
    
    input:
        training_set: the set of features of the data from which to build our model
        model_type: the scikit-learn model of choice given by user input parameter
        
    output:
        trained model fit to the features of the data
    '''
    
    # create a model variable
    model = None
    
    if model_type == 'latent dirichlet allocation':
        # instantiate model with default hyperparameter settings
        model = LatentDirichletAllocation(n_topics=5)
        
        if cross_validate:
            # create parameter distributions
            param_distro = {}
            
            # create random grid search object
            model = RandomizedSearchCV(estimator=model, param_distributions=param_distro, n_iter=num_iters, n_jobs=-1, verbose=True)
            
            print '\n', '... performing cross-validation', '\n'
            
            # cross-validate the model
            model.fit(data)
        else:
            # fit the vanilla model to the data
            model.fit(data)
        
    elif model_type == 'non-negative matrix factorization':
        # instantiate model with default hyperparameter settings
        model = NMF(n_components=5)
        
        if cross_validate:
            # create parameter distributions
            param_distro = {}
            
            # create random grid search object
            model = RandomizedSearchCV(estimator=model, param_distributions=param_distro, n_iter=num_iters, n_jobs=-1, verbose=True)
            
            print '\n', '... performing cross-validation', '\n'
            
            # cross-validate the model
            model.fit(data)
        else:
            # fit the vanilla model to the data
            model.fit(data)
    
    else:
        raise NotImplementedError
        
    # return the fitted / cross-validated model
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
    
    
def plot_predictions_ground_truth(predictions, ground_truth, handle, save_title):
    '''
    This function takes in the fitted model's predictions and the ground truth targets,
    and plots the two on a line graph, to show the user the relative accuracy of the predictions.
    
    input:
        prediction: the fitted model's predictions on the test data
        ground_truth: the actual value of the engagement parameters we fitted the model to
        
    output:
        A matplotlib line graph of both the predictions and the ground truth values
    '''
    
    # creating the plot
    plt.plot(range(len(predictions[:150])), np.round(predictions[:150]), 'r', label='Model Predictions')
    plt.plot(range(len(ground_truth[:150])), ground_truth[:150], 'b', label='Ground Truth Values')
    plt.legend()
    plt.title('@' + handle + ': Predictions vs. Ground Truth on Test Dataset')
    plt.xlabel('Tweet Index')
    plt.ylabel('Predictions vs. Ground Truth')
    
    # save the plot
    plt.savefig('../plots/regression/' + save_title + '.png')
    
    # showing the plot
    plt.show()
    
    
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
    
 
def print_top_words(model, feature_names, n_top_words):
    '''
    Function borrowed from scikit-learn documentation to print the top words from 
    each of the topics.
    '''
    
    # for each topic and its index
    for topic_idx, topic in enumerate(model.components_):
        # pint the topic index
        print "Topic #%d:" % (topic_idx + 1)
        # print the n_top_words top words
        print " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        
        
def remove_outliers(features, targets):
    '''
    This pre-processing step removes tweets which have more than 3 standard deviations
    more than the mean total engagement.
    
    input:
        tweets: set of tweets, possibly containing grievious outliers
        
    output:
        same tweets dataset, minus the outliers
    '''
    
    # compute mean, standard deviation
    mean = np.mean(targets, axis=0)
    std = np.std(targets, axis=0)
    
    # return features, targets who lie within 3 standard deviations of the mean
    # note: we only use the upper end of the interval, since the low end is
    # typically negative (and therefore impossible)
    idxs = np.where(targets <= (mean + 2 * std))  
    return features[idxs], targets[idxs]
        
        
def remove_link(text):
    '''
    Function borrowed to remove links in tweets.
    '''
    
    # https regex
    regex = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    pattern = re.compile(regex)
    return pattern.sub('', text)
    
    
