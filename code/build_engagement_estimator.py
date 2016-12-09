'''
Call this script to build a model of engagement based on a tweet
dataset

sample command: python build_engagement_estimator.py

author: Dan Saunders (djsaunde@umass.edu)
'''


# importing Twitter API python wrapper, argument-parser, and csv for csv manipulation
import tweepy, csv, os, sys, cPickle as pickle
# import helper methods
from util import *


############################
# GET PARAMETERS FROM USER #
############################

print '\n'

print 'Currently stored tweet datasets:', '\n'

for handle in os.listdir('../data'):
    print handle[:-11], '|',
    
print '\n'

# get Twitter handle to build an estimator on...
handle = raw_input('Enter Twitter handle for regression task: ')
print '\n'

# check to see if this handle exists
if handle + '_tweets.csv' not in os.listdir('../data/'):
    # ask the user if he / she would like to get this
    data_gather = raw_input('You do not have this data. Would you like to gather it? (y or Enter / n) ')
    if data_gather in ['y', '']:
        os.system('python get_data.py ' + handle)
    else:
        sys.exit() 

# get training / testing percentage...
train_split = int(raw_input('Enter percent data to use for training: '))
assert train_split > 0 and train_split <= 100
test_split = 100 - train_split
print '\n'

# get regression model to use to build estimator...
regression_method = raw_input('Enter regression model (linear regression [1], support vector regression [2], decision tree regression [3], neural network regression [4]) to use for engagement estimation: ')

if regression_method == '1':
    regression_method = 'linear regression'
elif regression_method == '2':
    regression_method = 'support vector regression'
elif regression_method == '3':
    regression_method = 'decision tree regression'
elif regression_method == '4':
    regression_method = 'neural network regression'
    
print '\n'
    
# extra parameters dictionary
params = {}    

# if neural network, get layer sizes
if regression_method == 'neural network regression':
    layer_sizes = tuple(int(layer_size) for layer_size in raw_input('Enter comma-separated list of layer sizes: ').split(','))
    params['layer_sizes'] = layer_sizes
    print '\n'

# get feature representation...
feature_repr = raw_input('Enter feature representation (count vectorizer [1], tfidf [2], hash vectorizer [3]) to use: ')

if feature_repr == '1':
    feature_repr = 'count vectorizer'
if feature_repr == '2':
    feature_repr = 'tfidf'
if feature_repr == '3':
    feature_repr = 'hash vectorizer'

print '\n'

# get parameter to perform regression on...
regress_param = raw_input('Enter parameter (likes [1], retweets [2], both [3]) to perform regression on: ')

if regress_param == '':
    regress_param = 'both'
elif regress_param == '1':
    regress_param = 'likes'
elif regress_param == '2':
    regress_param = 'retweets'
elif regress_param == '3':
    regress_param = 'both'
    
print '\n'

# get cross validation flag...
if regression_method in ['support vector regression', 'decision tree regression', 'neural network regression']:
    cv_flag = raw_input('Random search cross validation (y) or standard initialization (n or hit Enter): ')
    print '\n'
else:
    cv_flag = ''

# branch according to cross validation flag...
if cv_flag == 'y':
    cross_validate = True
    num_iters = int(raw_input('Enter number of iterations to run random search CV: '))
    print '\n'
elif cv_flag == 'n' or cv_flag == '':
    cross_validate = False
    num_iters = 0
else:
    raise ValueError('Need (y / n / Enter) input')
    
    
###########################################
# GET FEATURES, BUILD THE MODEL, EVALUATE #
###########################################    

# parse the scraped dataset into its feature representation
features, targets, feature_extractor = get_features_and_targets_regression(handle + '_tweets.csv', regress_param, feature_repr)

# remove outliers from the features, targets based on target values (can't figure out both implementation)
if regress_param != 'both':
    features, targets = remove_outliers(features, targets)

# packing together
dataset = (features, targets)

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, train_split, test_split)

# print out shapes of data
print 'training features shape:', train_data[0].shape, '\n'
print 'training labels shape:', train_data[1].shape, '\n'
print 'testing features shape:', test_data[0].shape, '\n'
print 'testing labels shape:', test_data[1].shape, '\n'

# build an estimator based on the training dataset
model = build_regression_model(train_data, regression_method, cross_validate, num_iters, params)

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# create a filename based on parameters
fname = handle + ' ' + str(train_split) + ' ' + str(test_split) + ' ' + regression_method + ' ' + feature_repr + ' ' + regress_param + ' ' + str(num_iters) + 'cv'

# print relevant information to the console
print '---> R^2 score on training set:', train_score, '\n'
print '---> R^2 score on test set:', test_score, '\n'
print '---> regression mean absolute error on training set:', train_mean_abs_error, '\n'
print '---> regression mean absolute error on test set:', test_mean_abs_error, '\n'
print '---> mean of regression parameter(s) value:', float(np.sum(targets)) / np.prod(targets.shape), '\n'
print '---> standard deviation of regression parameter(s) value', np.std(targets), '\n'

# show the plot of predicted engagement vs. ground truth
plot_predictions_ground_truth(model.predict(test_data[0]), test_data[1], handle, fname)

# print out model parameters
print 'Model and parameters:', model
print '\n'

# prompt the user, ask whether to save the model
save = raw_input('Do you want to save this model (y or Enter / n)? ')
print '\n'

if save == 'y' or save == '':
    with open('../models/regression/' + fname + '.p', 'wb') as f:
        pickle.dump((model, feature_extractor), f)


