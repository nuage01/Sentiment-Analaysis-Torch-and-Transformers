#!/usr/bin/python

"""model made with the help of
samer.azar@dataimpact.io
ahmad.nadar@dataimpact.io
Data Scientists at https://dataimpact.io/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import gensim, nltk, re

from tensorflow.python.keras import regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Bidirectional, Conv1D, CuDNNLSTM, Dense, Dropout, Embedding, LSTM
from tensorflow.python.keras.layers import normalization, Input, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

def create_tokenizer(line):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(line)
    
    return tokenizer

def encode_docs(tokenizer, max_length, docs):

    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen = max_length, padding = 'post')
    
    return padded

def encode_docs_new_vocab(sp, max_length, docs):
    
    encoded =  [sp.EncodeAsIds(doc) for doc in docs]
    padded = pad_sequences(encoded, maxlen = max_length, padding = 'post')
    
    return padded

def f1(y_true, y_pred):    
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def generate_data(df, mean_length, ratio, token=None, sp=None):
    
    # split dataframe into singles dataframes for each rating score
    data_1 =  df.loc[lambda df: df['review_rating'] == 1]
    data_2 =  df.loc[lambda df: df['review_rating'] == 2]
    data_3 =  df.loc[lambda df: df['review_rating'] == 3]
    data_4 =  df.loc[lambda df: df['review_rating'] == 4]
    data_5 =  df.loc[lambda df: df['review_rating'] == 5]
    
    # spliting each score dataframe into two dataframes set by a ratio
    data_val_1 = data_1[:int(ratio*len(data_1))]
    data_train_1 =  data_1[int(ratio*len(data_1)):]

    data_val_2 = data_2[:int(ratio*len(data_2))]
    data_train_2 =  data_2[int(ratio*len(data_2)):]

    data_val_3 = data_3[:int(ratio*len(data_3))]
    data_train_3 =  data_3[int(ratio*len(data_3)):]

    data_val_4 = data_4[:int(ratio*len(data_4))]
    data_train_4 =  data_4[int(ratio*len(data_4)):]

    data_val_5 = data_5[:int(ratio*len(data_5))]
    data_train_5 =  data_5[int(ratio*len(data_5)):]
    
    # concat dfs split by ratio
    train_x = pd.concat([data_train_1, data_train_2,data_train_3,  data_train_4, data_train_5])
    val_x = pd.concat([data_val_1, data_val_2,data_train_3, data_val_4, data_val_5])
    
    # setting positifs 1 for rating >3
    train_x['score'] = train_x['review_rating'].apply(lambda x: 1 if x > 3 else 0)
    val_x['score'] = val_x['review_rating'].apply(lambda x: 1 if x > 3 else 0)
    
    train_y = train_x['score'].values
    val_y = val_x['score'].values
    
    #applying categorical from keras
    y_train =  to_categorical(train_y)
    y_val = to_categorical(val_y)
    
    # choosing tokenization by word or bpe
    if sp == None:
        X_train = encode_docs(token, mean_length, train_x['review_body'])
        X_val = encode_docs(token, mean_length, val_x['review_body'])
    else:
        X_train = encode_docs_new_vocab(sp, mean_length, train_x['review_body'])
        X_val = encode_docs_new_vocab(sp, mean_length, val_x['review_body'])
    
    return X_train, y_train, X_val, y_val

def ml_model_score(vocab_size, input_length, dimension):
    
    embedding_layer = Embedding(vocab_size, dimension, input_length=input_length)
    sequence_input = Input(shape=(input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(64, return_sequences=False))(embedded_sequences)
    x = Dropout(0.4)(x)
    x = Dense(64,  activation = 'relu')(x)
    x = Dropout(0.3)(x)

    output_tensor = Dense(2, activation = 'softmax')(x)
    
    return Model(sequence_input, output_tensor)

def ml_model_topics(vocab_size, input_length, dimension):
    
    embedding_layer = Embedding(vocab_size, dimension, input_length=input_length)
    sequence_input = Input(shape=(input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(64, return_sequences=False))(embedded_sequences)
    x = Dropout(0.4)(x)
    x = Dense(64,  activation = 'relu')(x)

    output_tensor = Dense(6, activation = 'sigmoid')(x)
    
    return Model(sequence_input, output_tensor)


def precision(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_pos / (predicted_pos + K.epsilon())
    return _precision

def recall(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    _recall = true_pos / (possible_pos + K.epsilon())
    return _recall


if __name__ == "__name__":
    pass
