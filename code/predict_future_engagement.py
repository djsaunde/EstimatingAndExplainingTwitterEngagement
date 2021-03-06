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
    
    for i, item in enumerate(os.listdir('../models/regression/')):
        print i+1, item
    
    print '\n'
    
    # get user input as to which model to use in making predictions
    model_idx = int(raw_input('Enter in the index (1, 2, 3, ...) of the model you wish to use to make predictions: '))
    
    print '\n', '... loading the model'
    
    # load the model from the pickled file
    model_name = os.listdir('../models/regression/')[model_idx-1]
    model, feature_extractor = pickle.load(open('../models/regression/' + model_name, 'rb'))
    
    # get regression parameter
    if 'likes' in model_name:
        regress_param = 'likes'
    elif 'retweets' in model_name:
        regress_param = 'retweets'
    else:
        regress_param = '(likes, retweets)'
    
    # create flag to cease using model
    done_model = False
    
    # enter into a loop until user specifies wanting to use a different model
    while not done_model:
    
        print '\n'
        
        # storing a list of tweets (to be entered)
        tweets = []
        
        while True:
            # get sample tweet from user input
            tweet = raw_input('Enter a tweet to make predictions on (or hit Enter to stop entering tweets): ')
            
            # if the user hit Enter only, break out of the tweet entering stage
            if tweet == '':
                break
            
            # otherwise, add the entered text into a list 
            tweets.append(tweet)
            
        # if the user entered no tweets, break out of the loop
        if len(tweets) == 0:
            continue
        
        # transform it into our feature space
        tweet_transform = feature_extractor.transform(np.array(tweets))
        
        # get the raw regression score(s) from the chosen model
        scores = model.predict(tweet_transform)
        
        # round each score to the nearest integer
        for i in range(len(scores)):
            scores[i] = int(round(scores[i]))
        
        print '\n', 'Making predictions on the entered tweets...', '\n'
        
        # print out tweet and corresponding predicted engagement
        for i in range(len(scores)):
            print tweets[i], ':', scores[i], regress_param
            
        print '\n'
        
        done_model = bool(int(raw_input('Do you want to choose a new model? (1 for yes, 0 for no) ')))
        
    del model
        
    print '\n'
    done = bool(int(raw_input('Do you want to quit? (1 for yes, 0 for no) ')))
    
print '\n'
    
    
    
