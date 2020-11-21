#########################
# Variables
#########################

covid_list = ['covid','virus', 'corona','ncov','sars','pandemic', 'new case', 'active case', 'social distanc', 'mask', 'ppe', 'quarantine']

rt_regex = "^rt"

#########################
# Preprocessing Functions
#########################

def preprocess(text, hashtags=False, join=False):
    """Strips out URLs, usernames, punctuation, and other unwanted text.
    Tweets are tokenzied and stop words removed. Removing hashtags and
    joining tokens back into a string is optional.
    
    Example - Creating a new column in DataFrame:
    
    df['new_col'] = df['full_text'].apply(lambda x: preprocess(x, hashtags=True))
    """
    text = text.lower()
    if hashtags:
        text = ' '.join(re.sub(r"\#\w*[a-zA-Z]+\w*","",text).split())
    text = ' '.join(re.sub("((www\.[\S]+)|(https?://[\S]+))","",text).split())
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",text).split())
    text = ' '.join(re.sub("^\n","",text).split())
    text = ' '.join(re.sub("^rt","",text).split())
    punc = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(punc)
    stops = [word for word in tokens if word not in stop_words]
    if join:
        stops = (' ').join(stops)
    return stops

def joiner(text):
    """Simple function to join a list together into one string"""
    string = (' ').join(text)
    return string

#########################
# Feature Functions
#########################

def covid_mention(text, synonyms=covid_list):
    """ Checks tweet for presence of any word from the synonyms list.
    Returns a binary if word is present. text must be lowercase
    
    Arguments:
        synonyms: A list object
    Exampe:
        df['covid_mention'] = df['full_text'].apply(covid_mention)
    """
    for term in synonyms:
        if term in text:
            return 1
        continue
    return 0

def is_retweet(text):
    """ Checks if tweet is a retweet.
    Returns a binary. text must be lowercase
    
    Exampe:
        df['is_retweet'] = df['full_text'].apply(is_retweet)
    """
    if re.match(rt_regex, text) is not None:
        return 1
    return 0

#########################

#########################
