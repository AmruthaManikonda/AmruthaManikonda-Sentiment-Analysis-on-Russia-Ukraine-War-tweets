# -*- coding: utf-8 -*-

import os
import pandas as pd

tweet_count = 300
text_query = "#UkraineRussiaWar"
since_date = "2022-03-05"
until_date = "2022-03-23"

os.system('snscrape --jsonl --max-results {} --since {} twitter-search "{} until:{}"> UkraineRussiaWar.json'.format(tweet_count, since_date, text_query, until_date))

UkraineRussiaWar = pd.read_json('UkraineRussiaWar.json', lines=True)

UkraineRussiaWar.head()

UkraineRussiaWar.to_csv('UkraineRussiaWar.csv', sep=',', index=False)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', None)

UkraineRussiaWar['content']

tweet_count = 300
text_query = "#UkraineUnderAttaсk"
since_date = "2022-03-05"
until_date = "2022-03-23"

os.system('snscrape --jsonl --max-results {} --since {} twitter-search "{} until:{}"> UkraineUnderAttaсk.json'.format(tweet_count, since_date, text_query, until_date))

UkraineUnderAttaсk = pd.read_json('UkraineUnderAttaсk.json', lines=True)

UkraineUnderAttaсk.head()

UkraineUnderAttaсk.to_csv('UkraineUnderAttaсk.csv', sep=',', index=False)

UkraineUnderAttaсk['content']

tweet_count = 300
text_query = "#RussianUkrainianWar"
since_date = "2022-03-05"
until_date = "2022-03-23"

os.system('snscrape --jsonl --max-results {} --since {} twitter-search "{} until:{}"> RussianUkrainianWar.json'.format(tweet_count, since_date, text_query, until_date))

RussianUkrainianWar = pd.read_json('RussianUkrainianWar.json', lines=True)

RussianUkrainianWar.head()

RussianUkrainianWar.to_csv('RussianUkrainianWar.csv', sep=',', index=False)

RussianUkrainianWar['content']

tweet_count = 300
text_query = "#RussiaUkraineConflict"
since_date = "2022-03-05"
until_date = "2022-03-23"

os.system('snscrape --jsonl --max-results {} --since {} twitter-search "{} until:{}"> RussiaUkraineConflict.json'.format(tweet_count, since_date, text_query, until_date))

RussiaUkraineConflict = pd.read_json('RussiaUkraineConflict.json', lines=True)

RussiaUkraineConflict.head()

RussiaUkraineConflict.to_csv('RussiaUkraineConflict.csv', sep=',', index=False)

RussiaUkraineConflict['content']

"""### Merging all the csv files. """

df = pd.concat(map(pd.read_csv, ['UkraineRussiaWar.csv','UkraineUnderAttaсk.csv','RussiaUkraineConflict.csv','RussianUkrainianWar.csv']), ignore_index=True)

df

df['content']

"""### Filtering English tweets """

from langdetect import detect

df['LanguageReview'] = df['content'].apply(detect)

df

df[['content', 'LanguageReview']]

df1 = df[df['LanguageReview'] == 'en']
df1

df1['content']

import warnings
warnings.filterwarnings("ignore")

"""### Cleaning & VADER """

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')

stop_words=stopwords.words('english')
df1['content'] = df1['content'].apply(lambda txt: ' '.join([word for word in txt.split() if word not in stop_words]))

df1['content'] = df1['content'].apply(lambda txt: sent_tokenize(txt))

df1['content'] = df1['content'].apply(lambda txt: ' '.join(txt))

def remove_links(text):
    # Remove any hyperlinks that may be in the text starting with http
    import re
    return re.sub(r"http\S+", "", text)
def remove_tags(text):
    #remove any tags that may be in the text starting with @
    import re
    return re.sub(r"@\S+", "", text)
def remove_hashtags(text):
    #remove any hashtags that may be in the text starting with #
    import re
    return re.sub(r"#\S+", "", text)

df1['cleaned_text'] = df1['content'].astype(str).apply(remove_links)
df1['cleaned_text'] = df1['cleaned_text'].astype(str).apply(remove_tags)
df1['cleaned_text'] = df1['cleaned_text'].astype(str).apply(remove_hashtags)

df1['cleaned_text']

sid = SentimentIntensityAnalyzer()

df1['score'] = df1['cleaned_text'].apply(lambda txt: sid.polarity_scores(txt))

df1

df1['negative'] = df1['score'].apply(lambda txt: txt['neg'])
df1['neutral'] = df1['score'].apply(lambda txt: txt['neu'])
df1['positive'] = df1['score'].apply(lambda txt: txt['pos'])
df1['compound'] = df1['score'].apply(lambda txt: txt['compound'])

df1

def polarity_score(compound):
    if compound > 0.05:
        return "positive"
    elif compound < -0.05:
        return "negative"
    elif compound >= -0.05 and compound < 0.05:
        return "neutral"

df1['sentiment'] = df1['compound'].apply(lambda val: polarity_score(val))
df1

df1[['cleaned_text','negative','neutral','positive','compound','sentiment']]

df1['sentiment'].value_counts()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import pandas as pd

import matplotlib.pyplot as plt
# % matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#', 'rt', 'amp', 'http', 'https', '/', '://', '_', 'co', 'russia','ukraine'])

series_tweets = df1['cleaned_text']
tweet_str = series_tweets.str.cat(sep = ' ')
list_of_words = [i.lower() for i in wordpunct_tokenize(tweet_str) if i.lower() not in stop and i.isalpha()]
wordfreqdist = nltk.FreqDist(list_of_words)
mostcommon = wordfreqdist.most_common(100)
print(mostcommon)

import re
# Remove punctuation
df1['text_processed'] = \
df1['cleaned_text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
df1['text_processed'] = \
df1['text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
df1['text_processed'].head()

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['z', 'us', 'mariupol', 'kyiv', 'eu'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = df1.text_processed.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)

print(data_words[:1][0][:30])

data_words

import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

corpus

from pprint import pprint
# number of topics
num_topics = 10
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

import pyLDAvis
import pyLDAvis.gensim_models
import pickle 

# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join('C:/Users/amrut/SMM/results'+str(num_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, 'C:/Users/amrut/SMM/results'+ str(num_topics) +'.html')
LDAvis_prepared

#https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0
#https://akladyous.medium.com/sentiment-analysis-using-vader-c56bcffe6f24



