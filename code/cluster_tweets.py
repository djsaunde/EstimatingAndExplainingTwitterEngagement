'''
Use this script to experiment with tweets clustering. We measure the silhouette score
on the total engagement on the tweet dataset (namely, both likes and retweets).

sample command: python cluster_tweets.py

author: Dan Saunders (djsaunde@umass.edu)
'''


# import libraries
import sys, os, cPickle as pickle
# import helper methods
from util import *
# sklearn imports
from sklearn.metrics import silhouette_score


############################
# GET PARAMETERS FROM USER #
############################

print '\n'

print 'Currently stored tweet datasets:', '\n'

for handle in os.listdir('../data'):
    print handle[:-11], '|',
    
print '\n'

# get Twitter handle to build an estimator on...
handle = raw_input('Enter Twitter handle for clustering task: ')
print '\n' 

# check to see if this handle exists
if handle + '_tweets.csv' not in os.listdir('../data/'):
    # ask the user if he / she would like to get this
    data_gather = raw_input('You do not have this data. Would you like to gather it? (y or Enter / n) ')
    if data_gather in ['y', '']:
        os.system('python get_data.py ' + handle)
    else:
        sys.exit()
        
# get clustering method from user input
clustering_method = raw_input('Enter method of clustering (agglomerative clustering [1], kmeans [2]): ')

if clustering_method == '1':
    clustering_method = 'agglomerative clustering'
elif clustering_method == '2':
    clustering_method = 'kmeans'

print '\n'

# if we need the number of clusters beforehand...
if clustering_method in ['kmeans']:
    num_clusters = int(raw_input('Enter number of clusters to use (0 for cross-validation): '))
    print '\n'
    
    if num_clusters == 0:
        cv_flag = 'y'
    else:
        cv_flag = 'n'
        
else:
    num_clusters = int(raw_input('Enter number of clusters to use: '))
    print '\n'
    
    cv_flag = 'n'
    
# get feature representation...
feature_repr = raw_input('Enter feature representation (count vectorizer [1], tfidf [2], hash vectorizer [3]) to use: ')

if feature_repr == '1':
    feature_repr = 'count vectorizer'
elif feature_repr == '2':
    feature_repr = 'tfidf'
elif feature_repr == '3':
    feature_repr = 'hash vectorizer'

print '\n'

# whether or not to use latent semantic analysis dimensionality reduction
use_lsa = int(raw_input('Use latent semantic analysis (1 for yes, 0 for no)? '))

print '\n'

lsa_components = None
# get number of LSA components to use
if use_lsa:
    lsa_components = int(raw_input('Enter number of components to use for latent semantic analysis: '))
    print '\n'

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

# parse the scraped dataset into its feature representation and get targets and feature extractor
features, targets, feature_extractor = get_features_and_targets_clustering(handle + '_tweets.csv', feature_repr, use_lsa, lsa_components)

# build clustering model
if clustering_method in ['kmeans', 'agglomerative clustering']:
    model = build_clustering_model(features, clustering_method, cross_validate, num_iters, num_clusters)
else:
    model = build_clustering_model(features, clustering_method, cross_validate, num_iters)

if clustering_method in ['kmeans']:
    # print relevant information to the console
    print '---> clustering score:', model.score(features), '\n'

# print out model parameters
print 'Model and parameters:', model, '\n'


################
# MODEL SAVING #
################

# prompt the user, ask whether to save the model
save = raw_input('Do you want to save this model (y or Enter / n)? ')
print '\n'

if save == 'y' or save == '':
    with open('../models/clustering/' + handle + ' ' + clustering_method + ' ' + feature_repr + ' ' + str(num_clusters)
                + ' ' + str(num_iters) + 'cv.p', 'wb') as f:
        pickle.dump((model, feature_extractor), f)

