import itertools
import os

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import re

from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


datapath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
so_tags = pd.read_csv(datapath+'/02-data/SO_Train.csv')


def _extract_top_tags(df, tagvar, topn = 20):

    tagcounts = df.groupby([tagvar])['Id'].nunique()

    return tagcounts.sort_values(ascending=False).head(topn).keys()


def _filter_df_on_top_tags(df, tagvar, toptaglist):

    return df[df[tagvar].isin(toptaglist)]


def _remove_punctuation_from_string(givenstring):

    return re.sub(r'[^\w\s\']','',givenstring)


def _remove_stopwords(givenstring, language):

    textlist = givenstring.lower().split()
    thestopwords = stopwords.words(language)

    cleantextlist = list(set(textlist) - set(thestopwords))

    return ' '.join(cleantextlist)


def _create_train_test_data(df, trainproportion = 0.7):

    toptaglist = _extract_top_tags(df, 'Tags')
    filtereddf = _filter_df_on_top_tags(df, 'Tags', toptaglist)

    filtereddf['cleanTitle'] = filtereddf['Title'].apply(_remove_stopwords,  args = ('english',))
    filtereddf['cleanTitle'] = filtereddf['cleanTitle'].apply(_remove_punctuation_from_string)

