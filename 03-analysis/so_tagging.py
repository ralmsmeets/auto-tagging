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


def _get_topn_words(df, textvar, topn = 15000):

    titlewords = pd.Series(df[textvar].to_string().split())
    titlewords_frequency = titlewords.value_counts(ascending = False)
    titlewords_reduced = titlewords_frequency[0:topn]

    return titlewords_reduced.keys()


def _filter_out_uncommon_words(givenstring, commonwords):

    filtered_string = [w for w in givenstring.split() if w in commonwords]

    return ' '.join(filtered_string)


def _create_train_test_data(df, trainproportion = 0.7):

    toptaglist = _extract_top_tags(df, 'Tags')
    filtereddf = _filter_df_on_top_tags(df, 'Tags', toptaglist)

    filtereddf['cleanTitle'] = filtereddf['Title'].apply(_remove_stopwords,  args = ('english',))
    filtereddf['cleanTitle'] = filtereddf['cleanTitle'].apply(_remove_punctuation_from_string)

    words_to_keep = _get_topn_words(filtereddf, 'cleanTitle', topn = 1000)
    filtereddf['cleanShortTitle'] = filtereddf['cleanTitle'].apply(_filter_out_uncommon_words, args = (words_to_keep,))

    filtereddf = filtereddf.sample(frac = 1, random_state = 666)

    trainsize = int(len(filtereddf) * trainproportion)
    train_titles = filtereddf['cleanShortTitle'][:trainsize]
    train_tags   = filtereddf['Tags'][:trainsize]

    tokenize = text.Tokenizer(char_level=False)
    tokenize.fit_on_texts(train_titles)
    x_train = tokenize.texts_to_matrix(train_titles)

    encoder = LabelEncoder()
    encoder.fit(train_tags)
    y_train = encoder.transform(train_tags)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)

    num_classes = np.max(y_train) + 1
    y_train = utils.to_categorical(y_train, num_classes)

    return x_train, y_train


def _run_keras_model(xtrain, ytrain, batchsize, epochs, numdims):

    model = Sequential()
    model.add(Dense(numdims[0], input_shape=(xtrain.shape[1],)))
    model.add(Dense(numdims[1], input_shape=(numdims[0],)))
    model.add(Dense(numdims[2], input_shape=(numdims[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model.fit(xtrain, ytrain,
                        batch_size = batchsize,
                        epochs = epochs,
                        verbose = 1,
                        validation_split=0.05)


so_tags = pd.read_csv(datapath+'/02-data/SO_Train.csv')
x_train, y_train = create_train_test_data(so_tags, trainproportion = 1.0)
history = _run_keras_model(x_train, y_train, 32, 2, [528, 216, 108])