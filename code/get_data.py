'''
This script will scrape data from Twitter, using the python-twitter Twitter API
wrapper. Default behavior (no parameters) will be to scrape some number of 
tweets from specific Twitter handles. With parameters specified (in the form
of strings), this script will get a certain number of tweets from specified Twitter
handles.
'''

# importing Twitter API python wrapper, argument-parser, and csv for csv manipulation
import tweepy, argparse, csv
# import helper method script
from util import *


# create ArgumentParser object for handling user input
parser = argparse.ArgumentParser(description='Get Twitter handles.')
# adding argument for Twitter handle(s); default is Chancellor Subbaswamy's Twitter handle
parser.add_argument('handles', metavar='handles', type=str, nargs='*', default=['KSubbaswamy'], help='a Twitter handle to scrape data from')
# getting the arguments from the command line
args = parser.parse_args()


# Twitter API credentials
consumer_key = 'cbsWwUz5VxAg83smXCRbKwSc3'
consumer_secret = 'rvkJSG28FyYvAfg3V0bb9pDUyVFYZfQRHKFsc2oLfeQUgVoLxg'
access_key = '18059052-qVRCCrL00ww3OaWbRHGWSuPwj8ndnml4dvWqQMciw'
access_secret = 'U72wUy67OOPoFfZfrmPDEGWf5AppyTP4jrzmCXExTrFn7'


# authorize Twitter and initialize tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
# get API object using OAth credentials
api = tweepy.API(auth)


# for each Twitter handle, grab as many tweets as possible
for handle in args.handles:
    
    # storing all tweets for this Twitter handle
    tweets = []
    
    # make a request for most recent tweets (this gets as many as 200 tweets)
    scraped_tweets = api.user_timeline(screen_name=handle, count=200)
    
    # remove all retweets from the scraped tweets
    not_retweets = remove_retweets(scraped_tweets)
    
    # store the requested tweets in the mapping under the handle
    tweets.extend(not_retweets)
    
    # get the ID of the oldest tweet in the list
    oldest_ID = tweets[-1].id - 1
    
    # scrape tweets until we can scrape no more
    while scraped_tweets:
        print 'getting tweets before', oldest_ID
        
        # we add the oldest ID parameter in with subsequent requests
        scraped_tweets = api.user_timeline(screen_name=handle, count=200, max_id=oldest_ID)
        
        # remove all retweets from the scraped tweets
        not_retweets = remove_retweets(scraped_tweets)
        
        # then, we add the newly scraped tweets to our dictionary for this handle
        tweets.extend(not_retweets)
        
        # we update the pointer to the oldest ID we've scraped so far
        oldest_ID = tweets[-1].id - 1
        
        print len(tweets), 'tweets downloaded so far'
    
    # transform the tweepy tweets into a 2D array that will populate a csv	
	csv_tweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8"), tweet.favorite_count, tweet.retweet_count] for tweet in tweets]
	
	# write to a csv file	
	with open('../data/%s_tweets.csv' % handle, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(['id', 'created_at', 'text', 'favorites', 'retweets'])
		writer.writerows(csv_tweets)
