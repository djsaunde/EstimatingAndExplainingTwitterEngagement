'''
Call this script to predict engagement on as-of-yet unseen tweets

sample command: python predict_future_engagement.py

author: Dan Saunders (djsaunde@umass.edu)
'''


# import python libraries
import os, cPickle as pickle
# import helper methods
from util import *


# create program cease execution flag
done = False

# enter into a loop until user specifies a stopping point
while not done:
    # list models for user to pick from
    print 'Choose a model to use in making predictions...'
    
    print '\n'
    
    for i, item in enumerate(os.listdir('../models')):
        print i+1, item
    
    print '\n'
    
    # get user input as to which model to use in making predictions
    model_idx = int(raw_input('Enter in the index (1, 2, 3, ...) of the model you wish to use to make predictions: '))
    
    print '\n', '... loading the model', '\n'
    
    # load the model from the pickled file
    model_name = os.listdir('../models/')[model_idx-1]
    model, feature_extractor = pickle.load(open('../models/' + model_name, 'rb'))
    
    # create flag to cease using model
    done_model = False
    
    # enter into a loop until user specifies wanting to use a different model
    while not done_model:
        
        # get sample tweet from user input
        tweet = raw_input('Enter a tweet to make predictions on: ')
        
        print tweet
        
        # transform it into our feature space
        tweet_transform = feature_extractor.transform(np.array([tweet]))
        
        print tweet_transform
        
        # get the raw regression score(s) from the chosen model
        scores = model.predict(np.array([tweet_transform]))
        
        # round each score to the nearest integer
        for i in range(len(scores)):
            scores[i] = int(round(scores[i]))
        
        print '\n'
        print 'Output from regression model: ', scores
        
        done_model = bool(int(raw_input('Do you want to choose a new model? (1 for yes, 0 for no) ')))
        
    print '\n'
    done = bool(int(raw_input('Do you want to quit? (1 for yes, 0 for no) ')))
    
    
    
