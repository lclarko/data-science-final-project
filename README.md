# data-science-final-project

## Purpose
The aim of this project is to investigate, quantify and model the impact of Covid-19 on the expressed sentiment of British Columbians on Twitter, in relation to provincial politics.

## Data
The data analyzed in this project was collected from Twitter over a period over several months, from August 14th, 2020 to November 19th, 2020.

Using popular third party Twitter API tools, [Tweepy](https://www.tweepy.org/) and [Twarc](https://github.com/DocNow/twarc), nearly four hundred thousand tweet IDs in relation to #bcpoli were collected and later rehydrated for analysis. The Twitter data used in this project was collected in accordace with the [Twitter developer terms and conditions](https://developer.twitter.com/en/developer-terms).

## Reproducibility
If you have a Twitter developer account, this entire project is reproducible. In accordance with [Twitter's data redistribution policy](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases), only the tweet IDs have been published. Using these IDs and your own [Twitter API keys](https://developer.twitter.com/en/docs/twitter-api), you can rehydrate the tweets via a third party tool, like [Twarc](https://github.com/DocNow/twarc).

## Project Folder Structure:
```
data-science-final-project/
├── LICENSE
├── README.md
├── data
│   ├── bc_covid_data.sav
│   ├── bcpoli_tweet_id_400k.txt
│   └── bcpoli_vader_labelled_tweets.sav
├── deploy
│   ├── readme.txt
│   └── streamlit.zip
├── models
│   ├── LogReg_GridCV_3C_87p_40kfeats.sav
│   ├── LogReg_model_3C_86p__40kfeats.sav
│   └── NBMultinomial_model_3C_83p_40kfeats.sav
├── notebooks
│   ├── preliminary_analysis
│   │   ├── preliminary_classification_unlabelled_tweets.ipynb
│   │   ├── preliminary_twitter_data_exploration.ipynb
│   │   └── readme.txt
│   ├── bc_covid_data.ipynb
│   ├── sentiment_scoring_400K_tweets.ipynb
│   ├── training_baseline_classifier.ipynb
│   └── twitter_data_exploration_400k_tweets.ipynb
└── scripts
    └── functions.py
```

## Tools Used in this Project

Data Processing
* [Pandas](https://pandas.pydata.org)

Sentiment Analysis
* [VADER](https://github.com/cjhutto/vaderSentiment)
* [scikit-learn](https://scikit-learn.org/stable/)

Topic Modelling
* [spaCy](https://spacy.io/models/en)
* [Gensim](https://radimrehurek.com/gensim/)

App Deployment
* [Streamlit](https://www.streamlit.io/)

Tweet Scraping
* [Raspberry Pi Zero](https://www.raspberrypi.org/blog/raspberry-pi-zero/)
                
