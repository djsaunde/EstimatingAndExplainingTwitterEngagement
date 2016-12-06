'''
Call this script to build a model of engagement based on a tweet
dataset

sample command: python build_engagement_estimator.py

author: Dan Saunders (djsaunde@umass.edu)
'''


# importing Twitter API python wrapper, argument-parser, and csv for csv manipulation
import tweepy, csv, os, sys, cPickle as pickle
# import helper method script
from util import *


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
regression_method = raw_input('Enter regression model (linear regression, support vector regression, neural network regression) to use for engagement estimation: ')
print '\n'

# get feature representation...
feature_repr = raw_input('Enter feature representation (count vectorizer, tfidf, hash vectorizer) to use: ')
print '\n'

# get parameter to perform regression on...
regress_param = raw_input('Enter parameter (likes, retweets, both) to perform regression on: ')
if regress_param == '':
    regress_param = 'both'
    
print '\n'

# get cross validation flag...
if regression_method in ['support vector regression', 'neural network regression']:
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
    

# parse the scraped dataset into its feature representation
features, targets, feature_extractor = get_features_and_targets_regression(handle + '_tweets.csv', regress_param, feature_repr)
dataset = (features, targets)

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, train_split, test_split)

# build an estimator based on the training dataset
model = build_regression_model(train_data, regression_method, cross_validate, num_iters)

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '---> R^2 score on training set:', train_score
print '\n'
print '---> R^2 score on test set:', test_score
print '\n'
print '---> regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print '---> regression mean absolute error on test set:', test_mean_abs_error
print '\n'


# print out model parameters
print 'Model and parameters:', model
print '\n'

# prompt the user, ask whether to save the model
save = raw_input('Do you want to save this model (y or Enter / n)? ')
print '\n'

if save == 'y' or save == '':
    with open('../models/' + handle + ' ' + str(train_split) + ' ' + str(test_split)
                    + ' ' + regression_method + ' ' + feature_repr + ' ' + regress_param
                    + ' ' + str(num_iters) + '.p', 'wb') as f:
        pickle.dump((model, feature_extractor), f)


