"""
Script which contains helper functions for the main script(s).

author: Dan Saunders (djsaunde@umass.edu)
"""


def remove_retweets(tweets):
    """
    Function to remove tweets that are retweeted from another Twitter handle.
    For now, this simply checks for the occurrence of 'RT'. I believe there is
    a more clever way to do this (check Twitter API).
    
    input:
        tweets: list of tweets (typically 200, sometimes less).
    
    output:
        list of tweets with retweets removed.
    """
    
    # maintain a list of normal tweets (not retweets, that is)
    not_retweets = []
    # loop through each tweet
    for tweet in tweets:
        # if the tweet doesn't contain the string literal 'RT'...
        if 'RT' not in tweet.text:
            # add it to the returned list
            not_retweets.append(tweet)
            
    return not_retweets
    
    
