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

####################
# LIKES REGRESSION #
####################

# parse the scraped dataset into its feature representation
dataset = parse_dataset(args[0] + '_tweets.csv', 'likes')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on test set
score = get_score(model, test_data)

# print relevant information to the console
print '\n'
print 'likes regression score on test set:', score

#######################
# RETWEETS REGRESSION #
#######################

# parse the scraped dataset into its feature representation
dataset = parse_dataset(args[0] + '_tweets.csv', 'retweets')

# get the training set and test set based on input split parameter
train_data, test_data = split_dataset(dataset, int(args[1]), int(args[2]))

# build an estimator based on the training dataset
model = build_model(train_data, args[3])

# get model score on test set
score = get_score(model, test_data)

# print relevant information to the console
print '\n'
print 'retweets regression score on test set:', score


