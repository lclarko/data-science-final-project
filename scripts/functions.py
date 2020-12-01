import pandas as pd
import numpy as np
import pdpipe as pdp
import pickle

import re
import string
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


#########################
# Variables
#########################

covid_list = ['covid','virus', 'corona','ncov','sars',
              'super spread', 'super-spread', 'pandemic',
              'epidemic', 'outbreak', 'new case', 'new death',
              'active case', 'community spread', 'contact trac',
              'social distanc','self isolat', 'self-isolat', 'mask',
              'ppe', 'quarantine', 'lockdown', 'symptomatic', 'vaccine',
              'bonnie', 'new normal', 'ventilator', 'respirator', 'travel restrictions', 'doyourpartbc']

rt_regex = '(?:^rt|^RT)' # Now case insensitive

bc_cov19_url = 'https://health-infobase.canada.ca/src/data/covidLive/covid19-download.csv'


#########################
# Data Collection
#########################

def get_covid_data(df_name='df_bc_covid', globe=False):
    """
    Downloads Covid-19 data from Canada Gov site
    Filters on British Columbia
    Return DataFrame with date as index
    """
    if globe:
        global df_bc_covid
    try:
        df_bc_covid = pd.read_csv(bc_cov19_url)
    except:
        print('Reading CSV from URL failed')
    else:
        df_bc_covid = df_bc_covid[df_bc_covid.prname == 'British Columbia']
        df_bc_covid = df_bc_covid.set_index('date').fillna(0)
        df_bc_covid.drop(['pruid','prname','prnameFR'], axis=1, inplace=True)
        return df_bc_covid

#########################
# Preprocessing Functions
#########################

def col_filter(df,cols=['created_at','user','full_text','retweet_count', 'retweeted_status']):
    """
    Filters full twitter DataFrame down to desired columns
    """
    df.set_index('id_str', inplace=True)
    df = df[cols]
    return df

def preprocess(text, hashtags=False, join=False, url=True, user=True, emo=False):
    """
    Strips out URLs, usernames, punctuation, and other unwanted text.
    Tweets are tokenzied and stop words removed. Removing hashtags, URLs, 
    user mentions and joining tokens back into a string is optional.
    
    Example - Creating a new column in DataFrame:
    
    df['new_col'] = df['full_text'].apply(lambda x: preprocess(x, url=True, join=True, emo=True))
    """
    text = text.lower()
    if hashtags:
        text = ' '.join(re.sub(r'\#\w*[a-zA-Z]+\w*','',text).split())
    if url:
        text = ' '.join(re.sub('((www\.[\S]+)|(https?://[\S]+))','',text).split())
    if user:
        text = ' '.join(re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)','',text).split())
    if emo:
        # Positive Emoji - Smile, Laugh, Wink,Love
        text = ' '.join(re.sub('(:\s?\)|:-\)|:-\)\)|;\)|\(\s?:|\(-:|:\’\))',' emopos ',text).split()) 
        text = ' '.join(re.sub('(:\s?D|:-D|x-?D|X-?D)',' emopos ',text).split()) 
        text = ' '.join(re.sub('(<3|:\*)',' emopos ',text).split()) 
        # Negative Emoji - Sad, Cry
        text = ' '.join(re.sub('(:\s?\(|:-\(|:\||\)\s?:|\)-:)',' emoneg ',text).split())
        text = ' '.join(re.sub('(:,\(|:\’\(|:"\()',' emoneg ',text).split())
    text = ' '.join(re.sub('^\n','',text).split())
    text = ' '.join(re.sub('amp;',' ',text).split())
    text = ' '.join(re.sub('^rt','',text).split())
    punc = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(punc)
    stops = [word for word in tokens if word not in stop_words]
    if join:
        stops = (' ').join(stops)
    return stops

def tf_preprocess(text):
    """
    This will be removed
    """
    text = ' '.join(re.sub('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)',' ',text).split())
    text = ' '.join(re.sub('((www\.[\S]+)|(https?://[\S]+))','',text).split())
    text = ' '.join(re.sub('amp;',' ',text).split())
    text = ' '.join(re.sub('^rt',' ',text).split())
    return text

def vader_preprocess(text):
    """
    Alternate tweet processing for VADER, which can handle punctuation and capitalization.
    VADER can handle URLs. URLs don't influence sentiment scores, so they are removed
    """
    text = ' '.join(re.sub('((www\.[\S]+)|(https?://[\S]+))','',text).split())
    text = ' '.join(re.sub('^\n','',text).split())
    text = ' '.join(re.sub('amp;','',text).split())
    text = ' '.join(re.sub('^rt','',text).split())
    return text

def extract_full_text(df):
    """
    Extracts and creates a new column for username from user column, which exists as a dictionary with many keys.
    Joins rt_full_text onto main DataFrame
    Usage:
        df = extract_full_text(df)
    """
    temp = df['retweeted_status'].apply(pd.Series)
    temp['rt_full_text'] = temp['full_text']
    df = df.join(temp['rt_full_text'])
    return df

def replace_retweet_text(df,check_col='full_text',rt_col='rt_full_text'):
    """
    If retweet, extract the full_text from retweeted status and replaces the truncated version.
    """
    df.loc[df[check_col].str.contains(rt_regex, regex=True), check_col] = df[rt_col]
    df[check_col] = df[check_col].astype('str')
    return df

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

def lemmatize_text(text):
    """
    Lemmatize text, or reduce words to their root. e.g. leaders -> leader
    
    Usage:
    df['lemma'] = df[COL].apply(lambda x: lemmatize_text(x))
    """
    text = ' '.join([char for char in text if char not in string.punctuation])
    text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return text


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

def top_ngrams(df, n=2, ngrams=10):
    """
    * Not generalizable in this form *
    * This works well, but is very inefficient and should be optimized or rewritten *
    Takes a preposcessed, tokenized column and create a large list.
    Returns most frequent ngrams

    Arguments:
        df = name of DataFrame with no_hashtags column (this will be generalizable in a future commit)
        n = number of words per grouping eg. 1, 2 or 3
        ngrams = Number of ngrams to return
        """
    word_list = preprocess(''.join(str(df['lemma'].tolist())))
    return (pd.Series(nltk.ngrams(word_list, n)).value_counts())[:ngrams]

def vader_analyze(text):
    """
    Returns the compound sentiment score from VADER analyzer.polarity_score
    """
    score = analyzer.polarity_scores(text)
    return score

def vader_score_to_series(df):
    """
    Combines several functions to return pos, neg, neu and compound VADER scores as new columns
    Requires vader_analyze and categorize
    Usage:
        df = vader_score_to_series(df)
    """
    df['vader_scores'] = df['vader_text'].apply(vader_analyze)
    df = df.join(df['vader_scores'].apply(pd.Series))
    df.drop(columns='vader_scores', inplace=True, axis=1)
    df['vader_label'] = df['compound'].apply(lambda x: categorize(x)).astype('int8')
    return df

def categorize(x, upper = 0.05,lower = -0.05):
    """
    Categorizes tweets into sentiment categories of 0, 2 and 4.
    Negative, Netral and Postive, respectively.
    0, 2 and 4 were chosen to compare against another model that calssifies this way.
    The upper and lower variables are standard thresholds from VADER Sentiment
    """
    if x < lower:
        return 0
    elif ((x > (lower+0.0001) and x < upper)):
        return 2
    else:
        return 4

#########################

#########################
