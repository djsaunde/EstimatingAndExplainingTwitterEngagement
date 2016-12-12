'''
Use this script to experiment with tweets clustering. We measure the silhouette score
on the total engagement on the tweet dataset (namely, both likes and retweets).

sample command: python cluster_tweets.py

author: Dan Saunders (djsaunde@umass.edu)
'''


# import libraries
import sys, os, scipy, cPickle as pickle
# import helper methods
from util import *
# sklearn imports
from sklearn.metrics import silhouette_score
# import word cloud object
from wordcloud import WordCloud


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
    
# create a filename based on parameters
fname = handle + ' ' + clustering_method + ' ' + feature_repr + ' ' + str(num_clusters) + ' ' + str(num_iters) + 'cv'
    

###########################################
# GET FEATURES, BUILD THE MODEL, EVALUATE #
###########################################

# parse the scraped dataset into its feature representation and get targets and feature extractor
features, targets, feature_extractor = get_features_and_targets_clustering(handle + '_tweets.csv', feature_repr)

# build clustering model
if clustering_method in ['kmeans']:
    model = build_clustering_model(features, clustering_method, cross_validate, num_iters, num_clusters)
elif clustering_method in ['agglomerative clustering']:
    model = build_clustering_model(features.toarray(), clustering_method, cross_validate, num_iters, num_clusters)
else:
    model = build_clustering_model(features, clustering_method, cross_validate, num_iters)

if clustering_method in ['kmeans']:
    # print relevant information to the console
    print '---> clustering score:', model.score(features), '\n'

# print out model parameters
print 'Model and parameters:', model, '\n'


########################
# CREATE WORD CLOUD :) #
########################

# transform dataset back into terms-per-document representation
orig_features = np.array(feature_extractor.inverse_transform(features))

# get text from each cluster
clstrs_text = []
for clstr in xrange(model.n_clusters):
    # get cluster indices    
    clstr_idxs = np.where(clstr == model.labels_)
    
    # add all text from the cluster to the list
    clstrs_text.append(' '.join([' '.join([' '.join(orig_features[clstr_idxs[i]][j]) for j in range(len(clstr_idxs[i]))]) for i in range(len(clstr_idxs))]))
    
# create a word cloud for each cluster
for i, text in enumerate(clstrs_text):
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=65).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    
    # save the plot
    plt.savefig('../plots/clustering/' + fname + 'cluster' + str(i) + '.png')
    
    # show the plot
    plt.show()
    

################
# MODEL SAVING #
################

# prompt the user, ask whether to save the model
save = raw_input('Do you want to save this model (y or Enter / n)? ')
print '\n'

if save == 'y' or save == '':
    with open('../models/clustering/' + fname + '.p', 'wb') as f:
        pickle.dump((model, feature_extractor), f)

