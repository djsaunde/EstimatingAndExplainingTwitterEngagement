'''
Use this script to experiment with tweets clustering.

author: Dan Saunders (djsaunde@umass.edu)
'''


# import libraries
import sys, os
# import helper methods
from util import *


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
clustering_method = raw_input('Enter method of clustering (hierarchical clustering, kmeans): ')
print '\n'

# if we need the number of clusters beforehand...
if clustering_method in ['kmeans']:
    n_clusters = int(raw_input('Enter number of clusters to use: '))
    print '\n'
    
# get feature representation...
feature_repr = raw_input('Enter feature representation (count vectorizer, tfidf, hash vectorizer) to use: ')
print '\n'

# get cross validation flag...
cv_flag = raw_input('Random search cross validation (y) or standard initialization (n or hit Enter): ')
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
    
    
# parse the scraped dataset into its feature representation
features, feature_extractor = get_features_and_targets_clustering(handle + '_tweets.csv', feature_repr)

# build clustering model
model = build_clustering_model(features, clustering_method, cross_validate, num_iters)

print model
print model.score(features)
    

