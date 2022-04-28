from re import S
from socket import if_nameindex
# from wordcloud import WordCloud
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import re

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd

# ------------------------- streamlit part starts ---------------
# main menu 
st.write("""# Twitter Sentiment Anylysis Application""")
st.text("Submitted for: NUST DARA DATA SCIENCE SCHOOL 2022")

st.write("""----""")

# ----------Streamlit Side bar -------------------
st.sidebar.header('Keywords Selection')

query = st.sidebar.text_input("Input Keywords below to search: ")
country = st.sidebar.text_input("Input Country to search from: ")
date = st.sidebar.date_input("Select Date: ")
limit = st.sidebar.number_input("Number of Tweets: ")

st.sidebar.button('search')

# ---------------------- ---------------


tweets=[ ]

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets)==limit:
        break
    else:
        tweets.append([tweet.content, tweet.date, tweet.username])

data = pd.DataFrame(tweets, columns=['Tweet','Date','User'])

# save data to a csv file
data.to_csv('scrapped.csv')


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
st.text(f"{positive} %")
st.text(f"{negative} %")
st.text(f"{neutral} %")
st.text(f"{polarity} %")

# plotting
st.header("""Pie chart of positive, Negative, and Neutral Sentiment""")

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

#------------------------- pie chart ends --------------------------

# CLEAN TWEETS
#Clean the text using a function created below

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Removed mentions
    text = re.sub(r'#', '', text) #Removed hashtags
    text = re.sub(r'RT[\s]+', '', text) #Remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text) #Remove the hyperlink
    text = re.sub(r':[\s]+', '', text) #Remove columns
    text = re.sub(r'\'[\s]+', '', text) #Remove apostrophe
    text = re.sub(r'\...+', '', text) #Remove dots
    
    return text

#Cleaning the text
data['Tweet'] = data['Tweet'].apply(clean_text)

# SUBJECTIVITY FUNCTIONS
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# ------------- Word Cloud ______________________________

#See sentiment distribution, using wordcloud
# fig, ax = plt.subplots()

# allWords = ' '.join([twts for twts in data['Tweet']])
# wordCloud = WordCloud(width = 800, height= 500, random_state=21, max_font_size = 119).generate(allWords)

# ax.imshow(wordCloud, interpolation = "bilinear")
# ax.axis('off')

# st.header("Sentiment distribution, using wordcloud")
# st.pyplot(fig)