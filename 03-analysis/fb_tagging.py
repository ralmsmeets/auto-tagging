import itertools
import os

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


datapath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

so_train = pd.read_csv(datapath+'/02-data/SO_Train.csv')

def _extract_top_tags(df, tagvar, topn = 20):

    tagcounts = df.groupby([tagvar])['Id'].nunique()

    return tagcounts.sort_values(ascending=False).head(topn).keys()


def _filter_df_on_top_tags(df, tagvar, toptaglist):

    return df[df[tagvar].isin(toptaglist)]


