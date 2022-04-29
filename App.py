
import re, os, sys, nltk

import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

import snscrape.modules.twitter as sntwitter
import pandas as pd

from re import S
from socket import if_nameindex
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import date
from helpers import clean_text, get_subjectivity, get_polarity


nltk.download('vader_lexicon')

# ------------------------- streamlit part starts ---------------
# main menu 
st.write("""# Twitter Sentiment Anylysis Application""")
st.text("NUST-DARA DATA SCIENCE SCHOOL 2022")
st.write("""By: Robson & Chisabi""")

st.write("""----""")

# ----------Streamlit Side bar -------------------
st.sidebar.header('Keywords Selection')
st.write("""---""")

start_date = st.sidebar.date_input("Start Date: ")
end_date = st.sidebar.date_input("End Date: ")

keyword = st.sidebar.text_input("Enter Keyword: ")



limit = st.sidebar.number_input("Number of Tweets: ")

st.sidebar.button('search')

# ---------------------- ---------------


tweets = []
query = f"Twitter (from: {keyword}) until:{end_date} since:{start_date}"

# load data
@st.cache
def load_data(keywords, limit):
  for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets)==limit:
      break
    else:
      tweets.append([tweet.content, tweet.date, tweet.username, tweet.id])



load_data(query, limit)

data = pd.DataFrame(tweets,columns=['Tweet', 'Tweet Datetime', 'Tweet User', 'Tweet ID'])

# -----------------------------------------------

numOfTweets = len(list(data["Tweet"]))

# calculate percentage
def calculate_percentage(part,whole):
  return 100 * (float(part)/float(whole))


positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

# loop through a kl
for tweet in list(data['Tweet']):
  # print(tweet)
  # append tweets to tweet list
  tweet_list.append(tweet)
  analysis = TextBlob(tweet)
  # get the polarity score
  score = SentimentIntensityAnalyzer().polarity_scores(tweet)

  neg = score['neg']
  neu = score['neu']
  pos = score['pos']
  comp = score['compound']

  polarity += analysis.sentiment.polarity

  # check for positive and negative colleration
  if neg > pos:
    negative_list.append(tweet)
    negative += 1

  elif pos > neg:
    positive_list.append(tweet)
    positive += 1

  elif pos == neg:
    neutral_list.append(tweet)
    neutral += 1
    


  
positive = calculate_percentage(positive, numOfTweets)
negative = calculate_percentage(negative, numOfTweets)
neutral = calculate_percentage(neutral, numOfTweets)
polarity = calculate_percentage(polarity, numOfTweets)

# format the data to one decimal place
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')
polarity = format(polarity, '.1f')

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)


st.header("Sentiment Summary")
st.text(f"Total Number of Tweets: {len(tweet_list)}")
st.text(f"positive number: {len(neutral_list)}") 
st.text(f"Negative number: {len(negative)}")
st.text(f"Neutral number: {len(positive_list)}")

st.header('Percentage-Wise')
st.text(f"Positive: {positive} %")
st.text(f"Negative: {negative} %")
st.text(f"Neutral: {neutral} %")
st.text(f"Polarity: {polarity} %")
st.write("""---""")
#_---------------------------------------------------

# plotting
st.header("""Pie chart of positive, Negative, and Neutral Sentiment""")
st.write("""---""")
# ------------------ pie chart ---------------------
fig, ax = plt.subplots()

labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']

sizes = [positive, neutral, negative]

colors = ['yellowgreen', 'blue','red']

patches, texts = plt.pie(sizes,colors=colors, startangle=90)

ax.legend(loc="upper left", labels=labels)

ax.set_title(f"Sentiment Analysis Result for keyword = {query} ")
ax.axis('equal')


#show pie chart
st.pyplot(fig)
st.write("""---""")
# ----------------------------------
data['Tweet'] = data['Tweet'].apply(clean_text)


st.header("Sentiment distribution, using wordcloud")
st.write("""---""")
from wordcloud import WordCloud
allWords = ' '.join([twts for twts in data['Tweet']])
wordCloud = WordCloud(width = 800, height= 500, random_state=21, max_font_size = 119).generate(allWords)

fig,ax = plt.suplots()
ax.imshow(wordCloud, interpolation = "bilinear")
ax.axis('off')
st.pylot(fig)

sid = SIA()
results = []

def get_sentiment(row, **kwargs):
    sentiment_score = sid.polarity_scores(row)
    positive_meter = round((sentiment_score['pos'] * 10), 2)
    negative_meter = round((sentiment_score['neg'] * 10), 2) 
    return positive_meter if kwargs['k'] == 'positive' else negative_meter


data['positive'] = data.Tweet.apply(get_sentiment, k='positive')
data['negative'] = data.Tweet.apply(get_sentiment, k='negative')
data['neutral'] = data.Tweet.apply(get_sentiment, k='neutral')
data['compound'] = data.Tweet.apply(get_sentiment, k='compound')

data['label'] = 0
data.loc[data['positive'] > 0.2, 'label'] = 1
data.loc[data['negative'] > 0.2, 'label'] = -1


st.header("Plot of Percentage Sentiment")
st.write("""---""")
sns.set(rc={'figure.figsize':(8,6)})

counts = data.label.value_counts(normalize=True) * 100
fig, ax = plt.subplots()

ax = sns.barplot(x=counts.index, y=counts)
ax.set(title="Plot of Percentage Sentiment")
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

st.pyplot(fig)
st.write("""---""")
st.header("Boxplot to see average values of the labels and the positivity")
st.write("""---""")

boxplot = data.boxplot(column=['positive','negative', 'label'], 
                     fontsize = 15,grid = True, vert=True,figsize=(8,5,))
ax.set_ylabel('Range')
st.plotly_chart(boxplot)
st.write("""---""")

st.write("""Check back later for bugs :()""")
