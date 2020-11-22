import pandas as pd
import numpy as np
import pdpipe as pdp

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#########################
# Variables
#########################

covid_list = ['covid','virus', 'corona','ncov','sars', 'super spread', 'super-spread', 'pandemic', 'epidemic', 'outbreak', 'new case', 'new death', 'active case', 'community spread', 'contact trac', 'social distanc','self isolat', 'self-isolat', 'mask', 'ppe', 'quarantine', 'lockdown', 'symptomatic', 'vaccine', 'bonnie', 'new normal']

rt_regex = '(^rt)|(^RT)'

#########################
# Preprocessing Functions
#########################

def preprocess(text, hashtags=False, join=False):
    """
    Strips out URLs, usernames, punctuation, and other unwanted text.
    Tweets are tokenzied and stop words removed. Removing hashtags and
    joining tokens back into a string is optional.
    
    Example - Creating a new column in DataFrame:
    
    df['new_col'] = df['full_text'].apply(lambda x: preprocess(x, hashtags=True))
    """
    text = text.lower()
    if hashtags:
        text = ' '.join(re.sub(r'\#\w*[a-zA-Z]+\w*','',text).split())
    text = ' '.join(re.sub('((www\.[\S]+)|(https?://[\S]+))','',text).split())
    text = ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)','',text).split())
    text = ' '.join(re.sub('^\n','',text).split())
    text = ' '.join(re.sub('^rt','',text).split())
    punc = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(punc)
    stops = [word for word in tokens if word not in stop_words]
    if join:
        stops = (' ').join(stops)
    return stops

def vader_preprocess(text):
    """
    Alternate tweet processing for VADER, which can handle punctuation and capitalization.
    """
    pass

def emoji_stringer(text):
    """
    Converts common positive and negative ASCII emotions to 'emopos and 'emoneg'
    """    
    # Positive Emoji - Smile, Laugh, Wink,Love
    text = ' '.join(re.sub('(:\s?\)|:-\)|;\)|\(\s?:|\(-:|:\’\))','emopos',text).split()) # add this :-))
    text = ' '.join(re.sub('(:\s?D|:-D|x-?D|X-?D)','emopos',text).split()) 
    text = ' '.join(re.sub('(<3|:\*)','emopos',text).split()) 
    # Negative Emoji - Sad, Cry
    text = ' '.join(re.sub('(:\s?\(|:-\(|:\||\)\s?:|\)-:)','emoneg',text).split())
    text = ' '.join(re.sub('(:,\(|:\’\(|:"\()','emoneg',text).split())
    return text

def joiner(text):
    """
    Simple function to join a list together into one string
    """
    string = (' ').join(text)
    return string

def lower_case(text):
    """
    Simple function to convert text to lowercase
    Used in pipeline as workaround
    """    
    return text.lower()


#########################
# Feature Functions
#########################

def extract_username(df):
    """
    Extracts and creates a new column for username from user column, which exists as a dictionary with many keys.
    """
    df['user_name'] = df['user'].apply(lambda x: x.get('screen_name'))
    return df

def covid_mention(text, synonyms=covid_list):
    """
    Checks tweet for presence of any word from the synonyms list.
    Returns a binary if word is present. text must be lowercase
    
    Arguments:
        synonyms: A list object
    Example: df['covid_mention'] = df['full_text'].apply(covid_mention)
    """
    for term in synonyms:
        if term in text:
            return 1
        continue
    return 0

def is_retweet(text):
    """
    Checks if tweet is a retweet. Test is case insensitive
    Returns a binary.
    
    Exampe:
        df['is_retweet'] = df['full_text'].apply(is_retweet)
    """
    if re.match(rt_regex, text) is not None:
        return 1
    return 0

def top_bigrams(df, n=10):
    """
    * Not generalizable in this form *
    Takes a preposcessed, tokenized column and create a large list.
    Returns most frequent bigrams

    Arguments:
        df = name of DataFrame with no_hashtags column (this will be generalizable in a future commit)
        n = number of bigrams to return
        """
    word_list = preprocess(''.join(str(df['no_hashtags'].tolist())))
    return (pd.Series(nltk.ngrams(word_list, 2)).value_counts())[:n]

def vader_scores(text):
    """
    Returns the compound sentiment score from VADER analyzer.polarity_score
    """
    score = analyser.polarity_scores(text)
    return score['compound']

#########################

#########################
