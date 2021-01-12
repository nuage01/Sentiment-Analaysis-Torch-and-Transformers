import numpy as np
import sys
from string import punctuation
from nltk.corpus import stopwords
import logging
from tqdm import tqdm
from pandas import DataFrame
import dateparser
import re
from calendar import monthrange
from dataclasses import dataclass
from typing import Any, Dict, List, Callable
import pandas as pd
import os, importlib
import ast, pickle, spacy
from multiprocessing import cpu_count, Pool
import en_core_web_sm

global stopwords
stopwords = set(stopwords.words('english'))
stopwords = [word for word in stopwords if word != 'not']
global TO_REMOVE
PROCESSING_DICT = pickle.load(open('./src/processing_dict.p', 'rb'))


def parse_date(x):
    partitioned = x.partition(' on ')

    if(partitioned[-1] == ''):
        return dateparser.parse(partitioned[0]).date()
    else:
        return dateparser.parse(partitioned[-1]).date()


def fix_date(d):
    d_split = d.split('-')
    year = d_split[0]
    month = d_split[1]
    day = monthrange(int(year), int(month))[1]
    return year + '-' + month + '-' + str(day)


TO_REMOVE = [x for x in punctuation]


def apply_on_df(*a, **kw):
    def apply(func):
        df = kw['df']
        column = kw['column']
        df[column] = df[column].apply(lambda x: func(x))
    return apply


def process_data(df, retailer, country):

    df.columns = map(str.lower, df.columns)

    if('title' not in df):
        df['title'] = ''

    df = df.rename(index=str, columns={"content": "review_body", "date": "review_date", "product_id": "asin", "productname":"product_name",
                                       "product_ean": "ean", "rating": "review_rating", "title": "review_title", 'crawl_date': 'pp_date', 'enseignename': 'retailer'})

    df = df[['review_body', 'review_title', 'asin',
             'pp_date', 'review_rating', 'review_date', 'product_name', 'retailer']]

    df = df.drop_duplicates(keep='first')
    df = df.dropna(subset=['review_body'])
    df = df.dropna(subset=['asin'])
    df = df.dropna(subset=['pp_date'])
    df = df.dropna(subset=['review_rating'])
    df = df.dropna(subset=['review_date'])
    df = df.dropna(subset=['product_name'])


    df['review_title'] = df['review_title'].str.lower()
    df['review_body'] = df['review_body'].str.lower()

    df['review_rating'] = df['review_rating'].apply(
        lambda x: int(x[0]) if isinstance(x, str) else x)
    df = df[df['review_rating'] != 0]

    df['review_title'].fillna('',  inplace=True)
    df['review_title'] = df['review_title'].apply(lambda x:
                                                  x.replace('none', ''))

    df['review_body'] = df['review_body'].apply(lambda x:
                                                x.replace(' [this review was collected as part of a promotion.]', ''))

    df['text_clean'] = df['review_body']
    df['title_clean'] = df['review_title']

    df['review_date'] = df['review_date'].apply(lambda x: parse_date(x))

    @apply_on_df(df=df, column='text_clean')
    def replace_char_with_white(x):
        for char in TO_REMOVE:
            x = x.replace(char, '')
        return x

    @apply_on_df(df=df, column='text_clean')
    def regex_sub(x):
        return re.sub(r'\s+', ' ', x)

    @apply_on_df(df=df, column='title_clean')
    def regex_sub(x):
        return re.sub(r'\s+', ' ', x)

    @apply_on_df(df=df, column='text_clean')
    def expand(words):
        new_text = []
        for word in words.split():
            if word in PROCESSING_DICT:
                new_text.append(PROCESSING_DICT[word])
            else:
                new_text.append(word)
        return ' '.join(new_text)

    @apply_on_df(df=df, column='text_clean')
    def deEmojify(inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')

    @apply_on_df(df=df, column='text_clean')
    def remove_num(text):
        return ''.join([i for i in text if not i.isdigit()])

    @apply_on_df(df=df, column='text_clean')
    def replace_stopword(x):
        new_x = []
        for w in x.split():
            if stopwords != w:
                new_x.append(w)
        return ' '.join(new_x)

    @apply_on_df(df=df, column='text_clean')
    def remove_one_character(x):
        new_x = []
        for w in x.split():
            if len(w) != 1:
                new_x.append(w)
        return ' '.join(new_x)

    df.to_csv(retailer.lower() + '_cleaned.csv', index=False)
    logging.info('Data cleaned !')
    print('Data cleaned')


if __name__ == "__main__":
    retailer = sys.argv[1]
    df = pd.read_csv(retailer + '_reviews.csv')
    print(df.head())
    process_data(df, retailer, 'UK')
    logging.info('data correctly processed')
