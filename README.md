# Estimating and Explaining Twitter Engagement

This work is for the CS 589 (Machine Learning) final project.


## Scraping Tweets from Twitter

If you to scrape tweets from a new Twitter account, or get up-to-date tweets on a currently stored tweet dataset, follow these steps.

- Navigate to the code/ directory

- Execute "python get\_data.py \[handles\]", where the handles argument is replaced by 0 or more Twitter handles (default behavior is to scrape tweets Chancellor Kumble Subbaswamy of UMass Amherst).


## Building Twitter Engagement Estimator

To estimate engagement (in the form of likes, retweets, replies, ...) on a held-out set of data, follow these instructions.

- Navigate to the code/ directory

- Execute "python build\_engagement\_estimator.py", and follow the prompts at the console to enter in parameters for the regression estimator: handle, training data (and thus, testing data) percentage, regression model, feature representation / preprocessing, regression value (say, likes or retweets), random search cross-validation vs. standard scikit-learn initialization of the chosen model, and the number of iterations for which to run the random search CV.

- Finally, you will be asked whether or not you'd like to save the model (say, to use in estimating future tweet engagement or doing the clustering / explaing portions). If you answer in the affirmative, the model will be pickled with a canonical filename, which will be used to retrieve the model for other portions of the project.

## Estimating Future Tweet Engagement

To predict engagament (in the form of likes, retweets, replies, ...) on a new tweet or set of tweets, follow these instructions.

- Navigate to the code/ directory

- Execute "python predict\_future\_engagement.py", and follow the prompts at the console to choose the saved regression model to make predictions on new tweets which you may enter. You can continue to do this, or choose a new model and continue to do this, until you tell the program to stop after a prediction.


## Clustering Tweets at Different Levels of Granularity

Follow these instructions to experiment with clustering tweets in an unlabeled way, at different levels of granularity (i.e., number of clusters). This portion may also be useful for unsupervised feature learning.

- Navigate to the code/ directory

- Execute "python cluster\_tweets.py", and follow the prompts at the console to specify a Twitter handle, clustering algorithm, feature representation, whether or not to use latent semantic analysis (a dimensionality technique), and potentially whether or not to do random search hyperparameter cross-validation.

## Explaining Popular Tweets

Follow these instructions to experiment with dimensionality reduction, explanation, 
and exploration of a set of tweets.

- TODO 
