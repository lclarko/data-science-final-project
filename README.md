# data-science-final-project

## Purpose
The aim of this prject is to investigate, quantify and model the impact of Covid-19 on the expressed sentiment of British Columbians on Twitter, in relation to provincial politics.

## Data
The data analyzed in this project was collected from Twitter over a period over several months, from August 14th, 2020 to November 19th, 2020.

Using the popular third party Twitter API tools, [Tweepy](https://www.tweepy.org/) and [Twarc](https://github.com/DocNow/twarc), nearly four hundred thousand tweet IDs in relation to #bcpoli were collected and later rehydrated for analysis. The Twitter data used in this project was collected in accordace with the [Twitter developer terms and conditions](https://developer.twitter.com/en/developer-terms).

## Reproducibility
If you have a Twitter developer account, this entire project is reproducible. In accordance with [Twitter's data redistribution policy](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases), only the tweet IDs have been published. Using these IDs and your own [Twitter API keys](https://developer.twitter.com/en/docs/twitter-api), you can rehydrate the tweets via a third party tool, like [Twarc](https://github.com/DocNow/twarc).

## Project Folder Structure:
```
data-science-final-project/
├── LICENSE
├── README.md
├── data
│   ├── bcpoli_tweet_id_400k.txt
├── notebooks
│   ├── preliminary_analysis
│   │   ├── preliminary_twitter_data_exploration.ipynb
│   │   └── readme.txt
│   └── twitter_data_exploration_400k_tweets.ipynb
└── scripts
    └── functions.py
```
