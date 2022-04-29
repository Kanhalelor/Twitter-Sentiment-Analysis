import re
from textblob import TextBlob


def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Removed mentions
    text = re.sub(r'#', '', text) #Removed hashtags
    text = re.sub(r'RT[\s]+', '', text) #Remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text) #Remove the hyperlink
    text = re.sub(r':[\s]+', '', text) #Remove columns
    text = re.sub(r'\'[\s]+', '', text) #Remove apostrophe
    text = re.sub(r'\...+', '', text) #Remove dots
    
    return text

# get subjectivity score
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# get plarity score
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# calculate percentage
def calculate_percentage(part,whole):
  return 100 * (float(part)/float(whole))
