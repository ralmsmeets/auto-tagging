import pandas as pd
import numpy as np
import os

datapath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

fb_train = pd.read_csv(datapath+'/02-data/FB_Train.csv')

def _extract_top_tags(df, tagvar, topn = 20):

    tagcounts = df.groupby([tagvar])['Id'].nunique()

    return tagcounts.sort_values(ascending=False).head(topn).keys()


def _filter_df_on_top_tags(df, tagvar, toptaglist):

    return df[df[tagvar].isin(toptaglist)]


