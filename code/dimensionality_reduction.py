'''
Use this script to do dimensionality reduction on a set of tweets, visualize
important aspects of the data, and explore what underlies the engagement on a 
certain tweet dataset.

sample command: python dimensionality_reduction.py

author: Dan Saunders (djsaunde@umass.edu)
'''


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
handle = raw_input('Enter Twitter handle for dimensionality reduction / visualization task: ')
print '\n'

# check to see if this handle exists
if handle + '_tweets.csv' not in os.listdir('../data/'):
    # ask the user if he / she would like to get this
    data_gather = raw_input('You do not have this data. Would you like to gather it? (y or Enter / n) ')
    if data_gather in ['y', '']:
        execfile('python get_data.py ' + handle)
    else:
        sys.exit() 

# get regression model to use to build estimator...
dimensionality_reduction_method = raw_input('Enter dimensionality reduction method (latent dirichlet allocation [1], non-negative matrix factorization [2]): ')
print '\n'

if dimensionality_reduction_method == '1':
    dimensionality_reduction_method = 'latent dirichlet allocation'
elif dimensionality_reduction_method == '2':
    dimensionality_reduction_method = 'non-negative matrix factorization'

# get feature representation...
feature_repr = raw_input('Enter feature representation (count vectorizer [1], tfidf [2], hash vectorizer [3]) to use: ')

if feature_repr == '1':
    feature_repr = 'count vectorizer'
if feature_repr == '2':
    feature_repr = 'tfidf'
if feature_repr == '3':
    feature_repr = 'hash vectorizer'

print '\n'


#############################################
# GET FEATURES, DO DIMENSIONALITY REDUCTION #
#############################################

# parse the scraped dataset into its feature representation
features, targets, feature_extractor = get_features_and_targets_dimensionality_reduction(handle + '_tweets.csv', feature_repr)

# build dimensionality reduction model
model = build_dimensionality_reduction_model(features, dimensionality_reduction_method)

# print out model parameters
print 'Model and parameters:', model, '\n'


#######################################
# DO VISUALIZATION / TOPIC EXTRACTION #
#######################################

# if we used LDA or NMF and used TF-IDF, do topic extraction
if dimensionality_reduction_method in ['latent dirichlet allocation', 'non-negative matrix factorization'] and feature_repr in ['count vectorizer', 'tfidf']:
    feature_names = feature_extractor.get_feature_names()
    print_top_words(model, feature_names, n_top_words=10)
    
print '\n'


    





