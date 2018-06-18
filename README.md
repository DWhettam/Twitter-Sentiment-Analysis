# Twitter_Sentiment_Analysis
Live graphing of twitter sentiment analysis. Based on [this](https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL) tutorial by sentdex

## Use
- Create a folder named "pickled_algs" in the project directory
- Run TextClassification.py to generate classifiers (saved as pickle files)
- Update twitter_stream.py to include your own keys and tokens (can be generated [here](https://apps.twitter.com/))
- Update twitter_stream.py to analyse sentiment of specified topic
- Run twitter_stream.py
- Run live_graphing.py

## Requirements
- nltk
- sklearn
- tweepy
- matplotlib
