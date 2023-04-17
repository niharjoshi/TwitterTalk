#!/usr/bin/env python
# coding: utf-8

# In[41]:


# Importing necessary libraries

# For data manipulation
import re
import pickle
import numpy as np
import pandas as pd
import glob

# For EDA and visualization
import seaborn as sns
sns.set(rc={'figure.figsize': (20,10)})
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff


# For pre-processing text
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from better_profanity import profanity
profanity.load_censor_words()

# For other output and NLP utilities
from bokeh.plotting import figure
from bokeh.io import output_file, show, output_notebook
from collections import Counter
import spacy
from spacy.util import compounding
from spacy.util import minibatch
from spacy import displacy
import gc
import os
import urllib
import csv
from tqdm import tqdm

# For modelling and sentiment analysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# # Data preparation

# ## Dataset initialization
# 
# #### Note: the total dataset size is 17 GB and cannot be processed without a GPU so for the sake of demonstration, we will use a subset of the data

# In[2]:


# Loading all CSV files from the data/ folder
l = [pd.read_csv(filename, index_col=0, compression='gzip', low_memory=False) for filename in glob.glob("../data/input/*.gzip")]

# Combining all CSV files into a single dataframe
df = pd.concat(l, axis=0)


# In[3]:


df.head()


# ## Data cleaning

# In[4]:


df.shape


# In[5]:


# Checking column datatypes
df.info()


# In[6]:


# Checking for empty values
df.isna().sum().sort_values(ascending=False)


# In[7]:


# Removing profanity from tweet texts
# df['text'] = df['text'].apply(lambda x: profanity.censor(x))


# #### Since one of the steps in our EDA process is to check tweets by location, we need to handle empty location values

# In[8]:


# Handling NaN values for location
df = df.dropna(subset=['location'])


# # EDA

# ### Tweets by language

# In[9]:


# Getting top 5 languages for tweets
df.language.value_counts()[:5]


# In[10]:


# Plotting barplot for visualization
fig = sns.barplot(x=df.language.value_counts()[:5].index, y=df.language.value_counts()[:5])
fig.set(title='Tweets by langauge', xlabel='Language', ylabel='Tweet count (order of 10)')
fig = fig.get_figure()

# Saving barplot to file
fig.savefig('../static/tweets_by_language.png', bbox_inches='tight')


# #### For sentiment analysis, we will only be using tweets in English

# In[11]:


# Extracting all tweets in English language
df_en = df[df.language == 'en'].drop('language', axis=1)


# #### We will sort the tweets based on retweet count to judge for popularity

# In[12]:


# Sorting tweets based on retweet count
sorted_tweets = df_en[['username', 'text', 'retweetcount', 'tweetid']].sort_values(by='retweetcount', ascending=False).reset_index()

# Getting top 10 most retweeted tweets
sorted_tweets[['username', 'text']].head(10)


# ### Conversation topics and most used words

# In[13]:


# Building stopwords set
stopwords_set = set(STOPWORDS)
stopwords_set = set(stopwords.words('english'))


# In[14]:


# Generating word cloud
wordcloud = WordCloud(background_color='white',
                      stopwords=stopwords_set,
                      max_words=300,
                      max_font_size=40,
                      scale=2,
                      random_state=42)
wordcloud.generate(str(df_en['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/top_conversation_topics_wordcloud.png')
plt.show()


# #### Since a few tweets were retweeted too many times, it seems like a better idea to build a word cloud from only unique tweets

# In[15]:


# Getting unique tweets
unique_tweets = df_en.drop_duplicates(subset=['text'])

# Building word cloud
wordcloud= WordCloud(background_color='white',
                    stopwords=stopwords_set,
                    max_words=300,
                    max_font_size=40,
                    scale=2,
                    random_state=42
                    ).generate(str(unique_tweets.sort_values(by='retweetcount').iloc[:20]['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/top_unique_conversation_topics_wordcloud.png')
plt.show()


# ### Tweets by location

# In[16]:


# Plotting barplot for visualization
fig = df_en.location.value_counts()[:10].plot.bar()
fig.set(title='Tweets by location', xlabel='Location', ylabel='Location count')
fig = fig.get_figure()

# Saving barplot to file
fig.savefig('../static/tweets_by_location.png', bbox_inches='tight')


# ### One of the points discussed during the project proposal was to check for tweets by new accounts and check for possible propaganda

# #### Let's extract tweets by newest accounts and check how the wordcloud changes

# In[17]:


time_cols = ['extractedts', 'tweetcreatedts', 'usercreatedts']

# Converting "user account created time" column to datetime
df_en['usercreatedts'] = pd.to_datetime(df_en['usercreatedts'])

# Sorting by youngest user account age
sort_by_userage= df_en.sort_values(by='usercreatedts', ascending=True)


# In[18]:


# Getting columns
columns = df_en.columns.to_list()

# Building word cloud
wordcloud = WordCloud(background_color='white',
                      stopwords=stopwords_set,
                      max_words=300,
                      max_font_size=40,
                      scale=2,
                      random_state=42
                     ).generate(str(sort_by_userage.iloc[:1000, columns.index('text')]))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/newest_accounts_conversation_topics_wordcloud.png')
plt.show()


# # Sentiment analysis
# 

# #### Note: due to computation limitations, we will be using only the first 10000 rows for demonstration purposes

# In[123]:


# Building dataframe for sentiment analysis
sentiment_df = df_en[['text']].iloc[:10000]

sentiment_df.head()


# ### We will be using RoBERTa for sentiment analysis
# #### Note: we will be using the CPU as our local machine does not have an NVIDIA GPU

# In[124]:


device = "cuda:0" if torch.cuda.is_available() else "cpu"


# In[125]:


# Initializing tokenizer
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Initializing model
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment').to(device)

# Assigning sentiment labels
labels = ['negative', 'neutral', 'positive']


# In[126]:


# Setting batch size
BATCH_SIZE = 10

# Getting sentiment scores for tweet texts
scores_all = np.empty((0, len(labels)))
text_all = sentiment_df['text'].to_list()
n = len(text_all)
with torch.no_grad():
    for start_idx in tqdm(range(0, n, BATCH_SIZE)):
        end_idx = min(start_idx + BATCH_SIZE, n)
        encoded_input = tokenizer(text_all[start_idx:end_idx], return_tensors='pt', padding=True, truncation=True).to(device)
        output = model(**encoded_input)
        scores = output[0].detach().cpu().numpy()
        scores = softmax(scores, axis=1)
        scores_all = np.concatenate((scores_all, scores), axis=0)
        del encoded_input, output, scores
        torch.cuda.empty_cache()

# Saving scores to sentiment dataframe
sentiment_df['negative'] = [i[0] for i in scores_all]
sentiment_df['neutral'] = [i[1] for i in scores_all]
sentiment_df['positive'] = [i[2] for i in scores_all]


# In[127]:


sentiment_df.head()


# # Emotion detection

# In[128]:


# Initializing tokenizer
emotion_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")

# Initializing model
emotion_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion").to(device)

# Assigning sentiment labels
labels = ['anger', 'joy', 'optimism', 'sadness']


# In[129]:


# Setting batch size
BATCH_SIZE = 10

# Getting emotion scores for tweet texts
scores_all = np.empty((0, len(labels)))
text_all = sentiment_df['text'].to_list()
n = len(text_all)
with torch.no_grad():
    for start_idx in tqdm(range(0, n, BATCH_SIZE)):
        end_idx = min(start_idx + BATCH_SIZE, n)
        encoded_input = tokenizer(text_all[start_idx:end_idx], return_tensors='pt', padding=True, truncation=True).to(device)
        output = emotion_model(**encoded_input)
        scores = output[0].detach().cpu().numpy()
        scores = softmax(scores, axis=1)
        scores_all = np.concatenate((scores_all, scores), axis=0)
        del encoded_input, output, scores
        torch.cuda.empty_cache()
    

# Saving scores to sentiment dataframe
sentiment_df['anger'] = [i[0] for i in scores_all]
sentiment_df['joy'] = [i[1] for i in scores_all]
sentiment_df['optimism'] = [i[2] for i in scores_all]
sentiment_df['sadness'] = [i[3] for i in scores_all]


# In[130]:


sentiment_df.head()


# In[131]:


# Saving all scores as a dataset
sentiment_df.to_csv("../data/output/roberta_scores.csv", index=False)


# # Sentiment and emotion analysis 

# ## Sentiment analysis

# In[132]:


# Reading previously generated RoBERTa scores
tweet_df = pd.read_csv("../data/output/roberta_scores.csv", lineterminator='\n')

tweet_df.head()


# In[133]:


# Adding a sentiment column to save overall sentiment
tweet_df.insert(4, "sentiment", '')

tweet_df.head(0)


# In[134]:


# Computing overall sentiment for each tweet
for i in range(len(tweet_df)):
  if tweet_df['negative'][i] > tweet_df['positive'][i] and tweet_df['negative'][i] > tweet_df['neutral'][i]:
    tweet_df['sentiment'][i] = 'negative'
  elif tweet_df['positive'][i] > tweet_df['negative'][i] and tweet_df['positive'][i] > tweet_df['neutral'][i]:
    tweet_df['sentiment'][i]= 'positive'
  else:
    tweet_df['sentiment'][i] = 'neutral'

tweet_df.head()


# In[135]:


# Removing +ve, -ve, neutral columns as we don't need them anymore
tweet_df.drop(['negative','positive','neutral'], axis=1, inplace=True)


# In[136]:


# Saving overall sentiments as a dataset
tweet_df.to_csv("../data/output/roberta_overall_sentiment.csv", index=False)


# ### Plot for overall sentiment of tweets

# In[137]:


# Plotting barplot for visualization
plt.figure(figsize = (8,7))
fig = sns.countplot(x="sentiment", data=tweet_df, palette='magma')
fig = fig.get_figure()

# Saving barplot to file
fig.savefig('../static/overall_tweet_sentiment.png', bbox_inches='tight')


# ### Word cloud for negative, neutral and positive sentiments

# In[139]:


tweet_neg = tweet_df.loc[tweet_df['sentiment']=='negative'].reset_index(drop=True)
tweet_net = tweet_df.loc[tweet_df['sentiment']=='neutral'].reset_index(drop=True)
tweet_pos = tweet_df.loc[tweet_df['sentiment']=='positive'].reset_index(drop=True)


# #### Negative sentiment word cloud

# In[140]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(tweet_neg['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/negative_sentiment_wordcloud.png')
plt.show()


# #### Neutral sentiment word cloud

# In[141]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(tweet_net['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/neutral_sentiment_wordcloud.png')
plt.show()


# #### Positive sentiment word cloud

# In[142]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(tweet_pos['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/positive_sentiment_wordcloud.png')
plt.show()


# ## Emotion analysis

# In[172]:


# Reading previously generated RoBERTa scores
emotion_df = pd.read_csv("../data/output/roberta_scores.csv", lineterminator='\n')

emotion_df.head()


# In[173]:


# Removing +ve, -ve, neutral columns as we don't need them anymore
emotion_df.drop(['negative', 'positive', 'neutral'], axis=1, inplace=True)

# Adding a sentiment column to save overall sentiment
emotion_df.insert(5, "emotion", '')

emotion_df.head(0)


# In[174]:


# Computing overall emotion for each tweet
for i in range(len(emotion_df)):
  if emotion_df['anger'][i] > emotion_df['joy'][i] and emotion_df['anger'][i] > emotion_df['optimism'][i] and emotion_df['anger'][i] > emotion_df['sadness'][i]:
    emotion_df['emotion'][i] = 'anger'
  elif emotion_df['joy'][i] > emotion_df['anger'][i] and emotion_df['joy'][i] > emotion_df['optimism'][i] and emotion_df['joy'][i] > emotion_df['sadness'][i]:
    emotion_df['emotion'][i]= 'joy'
  elif emotion_df['optimism'][i] > emotion_df['anger'][i] and emotion_df['optimism'][i] > emotion_df['joy'][i] and emotion_df['optimism'][i] > emotion_df['sadness'][i]:
    emotion_df['emotion'][i]= 'optimism'
  else:
    emotion_df['emotion'][i] = 'sadness'

emotion_df.head(10)


# In[175]:


# Removing +anger, joy, optimism, sadness columns as we don't need them anymore
emotion_df.drop(['anger','joy','optimism','sadness'], axis=1, inplace=True)


# In[176]:


# Saving overall emotions as a dataset
emotion_df.to_csv("../data/output/roberta_overall_emotion.csv", index=False)


# ### Plot for overall emotion of tweets

# In[177]:


# Plotting barplot for visualization
plt.figure(figsize = (8,7))
fig = sns.countplot(x="emotion", data=emotion_df, palette='magma')
fig = fig.get_figure()

# Saving barplot to file
fig.savefig('../static/overall_tweet_emotion.png', bbox_inches='tight')


# ### Word cloud for anger, joy, optimism and sadness emotions

# In[178]:


emotion_anger = emotion_df.loc[emotion_df['emotion']=='anger'].reset_index(drop=True)
emotion_joy = emotion_df.loc[emotion_df['emotion']=='joy'].reset_index(drop=True)
emotion_opt = emotion_df.loc[emotion_df['emotion']=='optimism'].reset_index(drop=True)
emotion_sad = emotion_df.loc[emotion_df['emotion']=='sadness'].reset_index(drop=True)


# #### Anger emotion word cloud

# In[179]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(emotion_anger['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/anger_emotion_wordcloud.png')
plt.show()


# #### Joy emotion word cloud

# In[180]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(emotion_joy['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/joy_emotion_wordcloud.png')
plt.show()


# #### Optimism emotion word cloud

# In[181]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(emotion_opt['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/optimism_emotion_wordcloud.png')
plt.show()


# #### Sadness emotion word cloud

# In[182]:


# Building word cloud
wordcloud = WordCloud(background_color='white',
                     stopwords = stopwords_set,
                      max_words = 300,
                      max_font_size = 40,
                      scale = 2,
                      random_state=42
                     ).generate(str(emotion_sad['text']))

# Displaying and saving word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('../static/sadness_emotion_wordcloud.png')
plt.show()

