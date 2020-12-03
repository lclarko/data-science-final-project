import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title='BC SenTweet',
                    layout='wide',
                    page_icon=':boat:')

def main():
    # Define site pages here
    pages = {
        'Daily Metrics': page_first,
        'Weekly Metrics': page_second,
        'Ad Hoc Analysis': page_third,
        'Original Tweets':  page_fourth
    }
    # Left Hand Nav Title
    st.sidebar.title('Navigation')

    # Define left hand nav options (radio or drop down)
    page = st.sidebar.radio('', tuple(pages.keys()))
   
    # Display the selected page
    pages[page]()

def page_first():

    st.title('#BCPoli vs Daily Covid-19 Metrics')
    
    # Twitter Data + Covid Data
    bc_covid = pd.read_pickle('tweets_with_covid_data_v2.sav')
    bc_covid = bc_covid.drop(columns=['Retweet Ratio','Daily Deaths'], axis=1)

    bc_daily_real = pd.read_pickle('tweets_with_covid_data_real_values.sav')
    bc_daily_real = bc_daily_real.drop(columns=['Retweet Ratio','Daily Deaths'], axis=1)

    # Topic Model Data
    daily_topics = pd.read_pickle('dominant_topics_daily.sav')
    daily_topics = daily_topics.drop(columns=['Document_No'], axis=1).rename(columns={'Dominant_Topic':'Dominant Topic Number','Topic_Perc_Contrib':'Topic Contribution', 'Keywords': 'Topic Keywords'})

    # Create traces using a list comprehension:
    data1_daily = [go.Scatter(
        x=bc_covid.index,
        y=bc_covid[col],
        name=str(col)) for col in bc_covid.columns]

    # create a layout, remember to set the barmode here
    layout1_daily = go.Layout(
        title = 'Covid-19 Cases in BC vs Daily Average Tweet Sentiment & Covid-19 Mentions',
        xaxis = dict(title = 'Date'),
        yaxis = dict(title = 'Standardized Range: 0-1'),
        hovermode = 'closest',
        width = 1100, height = 550
    )

    fig1_daily = go.Figure(data=data1_daily, layout=layout1_daily)

    if st.checkbox('Chart Annotation'):
        fig1_daily.add_annotation(x='2020-10-14', y=1,
                text="Leaders' Debate",
                ax=0,
                ay=-40,
                showarrow=True,
                arrowhead=1)

        fig1_daily.add_annotation(x='2020-09-21', y=0.5219763,
                text='Election Announced',
                ax=0,
                ay=-50,
                showarrow=True,
                arrowhead=1)   

        fig1_daily.add_annotation(x='2020-10-25', y=1,
                text="NDP Victory",
                ax=0,
                ay=-20,
                showarrow=True,
                arrowhead=1)   

        fig1_daily.add_annotation(x='2020-10-26', y=0.8105159,
                text='Covid Daily Record',
                ax=60,
                showarrow=True,
                arrowhead=1)
        fig1_daily.add_annotation(x='2020-11-29', y=1,
                text='Covid Daily Record',
                ax=-60,
                showarrow=True,
                arrowhead=1)

    if st.checkbox('Show Chart', True):
        # create a fig from data & layout, and plot the fig.
        st.markdown('## Covid-19 Cases in BC vs Daily Average Tweet Sentiment & Covid-19 Mentions')
        st.plotly_chart(fig1_daily)

    if st.checkbox('Show Chart data'):
        st.markdown('Raw data')
        st.dataframe(bc_daily_real)

    if st.checkbox('Show Topic data'):
        st.dataframe(daily_topics)

#########################################################################################################

def page_second():
    st.title('Weekly #BCPoli vs Covid-19 Metrics')
    # Load labelled tweets and covid data
    bc_covid_weekly = pd.read_pickle('tweets_weekly_covid_data_v2.sav')
    df_weekly = bc_covid_weekly.drop(columns=['Retweet Ratio','Daily Deaths'], axis=1)

    if st.checkbox('Show Chart Data'):
        st.subheader('Raw data')
        st.write(bc_covid_weekly)

    # create traces using a list comprehension:
    data = [go.Scatter(
        x=df_weekly.index,
        y=df_weekly[col],
        name=str(col)) for col in df_weekly.columns]

    # create a layout, remember to set the barmode here
    layout = go.Layout(
        title = 'Daily Covid-19 Cases in BC vs Daily Average Tweet Sentiment',
        xaxis = dict(title = 'Date'),
        yaxis = dict(title = 'Normalized Sentiment'),
        hovermode = 'closest',
        width = 1000, height = 500
    )
    fig = go.Figure(data=data, layout=layout)
    if st.checkbox('Show Chart', True):
        # create a fig from data & layout, and plot the fig.
        st.plotly_chart(fig)
    
#########################################################################################################

def page_third():

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    #from nltk.sentiment import SentimentAnalyzer

    covid_list = ['covid','virus', 'corona','ncov','sars',
              'super spread', 'super-spread', 'pandemic',
              'epidemic', 'outbreak', 'new case', 'new death',
              'active case', 'community spread', 'contact trac',
              'social distanc','self isolat', 'self-isolat', 'mask',
              'ppe', 'quarantine', 'lockdown', 'symptomatic', 'vaccine',
              'bonnie', 'new normal', 'ventilator', 'respirator', 'travel restrictions', 'doyourpartbc']

    def main():
        st.markdown("""
                    # Ad Hoc Sentiment Analysis  
                    """
        )
        sentimental_analysis_component()

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
                return 'Yes'
            continue
        return 'No'

    def sentimental_analysis_component():
        """
        Ad Hoc Sentiment Analysis of a user input OR a Tweet ID lookup        
        """
        sentence = st.text_area("Enter Text to Analyze:")
        if st.button("Submit"):
            result = sentiment_analyzer_scores(sentence)
            st.success(result)

        #if st.checkbox('Lookup Twitter Status', True):
        id_input = st.text_area("Enter Tweet ID to Analyze:")
        st.markdown(' e.g. 1333434829438906376 or 1257038775785422848')

        # Modules for twitter API
        import tweepy 
        import os
        
       # API Keys
        consumer_key = os.environ.get('LHL_TWITTER_CONSUMER_KEY')
        consumer_secret = os.environ.get('LHL_TWITTER_CONSUMER_SECRET')
        access_token = os.environ.get('LHL_TWITTER_ACCESS_TOKEN')
        access_token_secret = os.environ.get('LHL_TWITTER_ACCESS_TOKEN_SECRET')
        
        # Auth type and API options
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth,wait_on_rate_limit=True)

        # Tweet ID to fetch
        id_ = [id_input]
        
        # API Call 
        statuses = api.statuses_lookup(id_, tweet_mode="extended")
        
        # API Response to variables
        for status in statuses:
            tweet_text = status.full_text
            tweet_user = status.user.screen_name
            covid_check = covid_mention(tweet_text.lower())

        if st.button("Analyze Tweet"):
            lookup_result = sentiment_analyzer_scores(tweet_text)
            st.markdown('## Tweet Sentiment Results')
            st.success(lookup_result)
            st.markdown(f'## Full Text:')
            st.success(f'{tweet_text}')

            st.markdown(f"""## Tweet Stats:
            Tweet ID:{id_}
            User: {status.user.screen_name}
            Created at: {status.created_at}
            Source: {status.source}
            Engagement:
                Retweets: {status.retweet_count}
                Favourited: {status.favorite_count}
            Pandemic Related: {covid_check}""")

    @st.cache
    def get_sentiment_analyzer():
        """
        Instantiates VADER SentimentIntensityAnalyzer
        """
        return SentimentIntensityAnalyzer()  # initialize it

    @st.cache
    def sentiment_analyzer_scores(sentence):
        """The sentiment scores of the sentence

        Arguments:
            sentence {[type]} -- [description]

        Returns:
            str -- [description]
        """
        score = get_sentiment_analyzer().polarity_scores(sentence)
        return 'Negative Score:', score['neg'], 'Neutral Score:', score['neu'], 'Positive Score:', score['pos'], 'Compound Score:', score['compound']

    main()

#########################################################################################################

def page_fourth():
    st.title('#BCPoli vs Covid-19 Metrics - Original Tweets Only')
    # Load labelled tweets and covid data
    bc_covid_daily_orig = pd.read_pickle('tweets_covid_daily_orig.sav')
    bc_covid_daily_orig = bc_covid_daily_orig.drop(columns=['Retweet Ratio','Daily Deaths'], axis=1)

    bc_daily_orig_real = pd.read_pickle('tweets_covid_daily_orig_real_values.sav')
    bc_daily_orig_real = bc_daily_orig_real.drop(columns=['Retweet Ratio','Daily Deaths'], axis=1)

    # Create traces using a list comprehension:
    data_orig = [go.Scatter(
        x=bc_covid_daily_orig.index,
        y=bc_covid_daily_orig[col],
        name=str(col)) for col in bc_covid_daily_orig.columns]

    # Create a layout, remember to set the barmode here
    layout_orig = go.Layout(
        title = 'Daily Covid-19 Cases in BC vs Daily Average Tweet Sentiment',
        xaxis = dict(title = 'Date'),
        yaxis = dict(title = 'Normalized Sentiment'),
        hovermode = 'closest',
        width = 1000, height = 500
    )
    fig2_daily_orig = go.Figure(data=data_orig, layout=layout_orig)

    if st.checkbox('Show Chart', True):
        # Create a fig from data & layout, and plot the fig.
        st.plotly_chart(fig2_daily_orig)

    if st.checkbox('Show Chart Data'):
        st.subheader('Raw data')
        st.write(bc_daily_orig_real)

if __name__ == "__main__":
    main()