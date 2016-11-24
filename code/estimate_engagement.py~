'''
Call this script to (1) build a model of engagement based on a tweet
dataset, and (2) predict engagement on as-yet unseen tweets.

sample command: python estimate_engagement.py 'BernieSanders' 80 20 'linear regression'

author: Dan Saunders (djsaunde@umass.edu)
'''

# importing Twitter API python wrapper, argument-parser, and csv for csv manipulation
import tweepy, argparse, csv
# import helper method script
from util import *


# create ArgumentParser object for handling user input
parser = argparse.ArgumentParser(description='Get Tweet training / test datasets and parameters.')
# adding all arguments (dataset, train / test split, model)
parser.add_argument('args', nargs=argparse.REMAINDER)

# getting the arguments from the command line
args = parser.parse_args().args


#################
# BAG OF WORDS  #
#################

print '\n'
print 'Trying bag of words feature representation...'

####################
# LIKES REGRESSION #
####################

# parse the scraped dataset into its feature representation
dataset = get_features(args[0] + '_tweets.csv', 'likes', 'count vectorizer')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '\n'
print 'likes regression score on training set:', train_score
print '\n'
print 'likes regression score on test set:', test_score
print '\n'
print 'likes regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print 'likes regression mean absolute error on test set:', test_mean_abs_error

#######################
# RETWEETS REGRESSION #
#######################

# parse the scraped dataset into its feature representation
dataset = get_features(args[0] + '_tweets.csv', 'retweets', 'count vectorizer')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '\n'
print 'retweets regression score on training set:', train_score
print '\n'
print 'retweets regression score on test set:', test_score
print '\n'
print 'retweets regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print 'retweets regression mean absolute error on test set:', test_mean_abs_error

#####################
# TDIDF TRANSFORMER #
#####################

print '\n'
print 'Trying TDIDF transform feature extraction...'

####################
# LIKES REGRESSION #
####################

# parse the scraped dataset into its feature representation
dataset = get_features(args[0] + '_tweets.csv', 'likes', 'tfidf')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '\n'
print 'likes regression score on training set:', train_score
print '\n'
print 'likes regression score on test set:', test_score
print '\n'
print 'likes regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print 'likes regression mean absolute error on test set:', test_mean_abs_error

#######################
# RETWEETS REGRESSION #
#######################

# parse the scraped dataset into its feature representation
dataset = get_features(args[0] + '_tweets.csv', 'retweets', 'tfidf')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '\n'
print 'retweets regression score on training set:', train_score
print '\n'
print 'retweets regression score on test set:', test_score
print '\n'
print 'retweets regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print 'retweets regression mean absolute error on test set:', test_mean_abs_error

######################
# HASHING VECTORIZER #
######################

print '\n'
print 'Trying hashing vectorizer feature extraction...'

####################
# LIKES REGRESSION #
####################

# parse the scraped dataset into its feature representation
dataset = get_features(args[0] + '_tweets.csv', 'likes', 'hashing vectorizer')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '\n'
print 'likes regression score on training set:', train_score
print '\n'
print 'likes regression score on test set:', test_score
print '\n'
print 'likes regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print 'likes regression mean absolute error on test set:', test_mean_abs_error

#######################
# RETWEETS REGRESSION #
#######################

# parse the scraped dataset into its feature representation
dataset = get_features(args[0] + '_tweets.csv', 'retweets', 'hashing vectorizer')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on training set, test set
train_score = get_score(model, train_data)
test_score = get_score(model, test_data)

# get model mean absolute error on training set, test set
train_mean_abs_error = get_mean_abs_error(model, train_data)
test_mean_abs_error = get_mean_abs_error(model, test_data)

# print relevant information to the console
print '\n'
print 'retweets regression score on training set:', train_score
print '\n'
print 'retweets regression score on test set:', test_score
print '\n'
print 'retweets regression mean absolute error on training set:', train_mean_abs_error
print '\n'
print 'retweets regression mean absolute error on test set:', test_mean_abs_error
